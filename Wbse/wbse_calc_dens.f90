!
! Copyright (C) 2015-2024 M. Govoni
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
! This file is part of WEST.
!
! Contributors to this file:
! Ngoc Linh Nguyen, Victor Yu
!
!---------------------------------------------------------------------
SUBROUTINE wbse_calc_dens(devc, drho, sf)
  !---------------------------------------------------------------------
  !
  ! This subroutine calculates the response charge density
  ! from linear response orbitals and ground state orbitals.
  !
  USE kinds,                  ONLY : DP
  USE cell_base,              ONLY : omega
  USE fft_base,               ONLY : dffts
  USE lsda_mod,               ONLY : nspin,lsda
  USE noncollin_module,       ONLY : npol
  USE pwcom,                  ONLY : npw,npwx,current_k,current_spin,isk,wg,ngk
  USE mp,                     ONLY : mp_sum,mp_bcast
  USE mp_global,              ONLY : my_image_id,inter_image_comm,inter_pool_comm,inter_bgrp_comm
  USE buffers,                ONLY : get_buffer
  USE westcom,                ONLY : iuwfc,lrwfc,nbnd_occ,n_trunc_bands
  USE fft_at_gamma,           ONLY : double_invfft_gamma
  USE distribution_center,    ONLY : kpt_pool,band_group
#if defined(__CUDA)
  USE wavefunctions_gpum,     ONLY : using_evc,using_evc_d,evc_work=>evc_d,psic=>psic_d
  USE wavefunctions,          ONLY : evc_host=>evc
  USE west_gpu,               ONLY : tmp_r
#else
  USE wavefunctions,          ONLY : evc_work=>evc,psic
#endif
  !
  IMPLICIT NONE
  !
  ! I/O
  !
  COMPLEX(DP), INTENT(IN) :: devc(npwx*npol,band_group%nlocx,kpt_pool%nloc)
  COMPLEX(DP), INTENT(OUT) :: drho(dffts%nnr,nspin)
  LOGICAL, INTENT(IN) :: sf
  !
  ! Workspace
  !
  INTEGER :: ir, ibnd, iks, nbndval, lbnd, dffts_nnr, iks_do
  REAL(DP) :: w1
#if !defined(__CUDA)
  REAL(DP), ALLOCATABLE :: tmp_r(:)
#endif
  INTEGER, PARAMETER :: flks(2) = [2,1]
  !
#if defined(__CUDA)
  CALL start_clock_gpu('calc_dens')
#else
  CALL start_clock('calc_dens')
#endif
  !
  dffts_nnr = dffts%nnr
  !
  !$acc kernels present(drho)
  drho(:,:) = (0._DP,0._DP)
  !$acc end kernels
  !
#if !defined(__CUDA)
  ALLOCATE(tmp_r(dffts%nnr))
#endif
  !
  DO iks = 1, kpt_pool%nloc  ! KPOINT-SPIN LOOP
     !
     IF(sf) THEN
        iks_do = flks(iks)
     ELSE
        iks_do = iks
     ENDIF
     !
     nbndval = nbnd_occ(iks_do)
     !
     ! ... Set k-point and spin
     !
     current_k = iks
     IF(lsda) current_spin = isk(iks)
     !
     ! ... Number of G vectors for PW expansion of wfs at k
     !
     npw = ngk(iks)
     !
     ! ... read GS wavefunctions
     !
     IF(kpt_pool%nloc > 1) THEN
        #if defined(__CUDA)
        IF(my_image_id == 0) CALL get_buffer(evc_host,lrwfc,iuwfc,iks_do)
          CALL mp_bcast(evc_host,0,inter_image_comm)
          CALL using_evc(2)
          CALL using_evc_d(0)
        #else
          IF(my_image_id == 0) CALL get_buffer(evc_work,lrwfc,iuwfc,iks_do)
          CALL mp_bcast(evc_work,0,inter_image_comm)
        #endif
     ENDIF
     !
     !$acc kernels present(tmp_r)
     tmp_r(:) = 0._DP
     !$acc end kernels
     !
     ! double bands @ gamma
     !
     DO lbnd = 1, band_group%nloc
        !
        ibnd = band_group%l2g(lbnd)+n_trunc_bands
        IF(ibnd < 1 .OR. ibnd > nbndval) CYCLE
        !
        w1 = wg(ibnd,iks_do)/omega
        !
        !$acc host_data use_device(devc)
        CALL double_invfft_gamma(dffts,npw,npwx,evc_work(:,ibnd),devc(:,lbnd,iks),psic,'Wave')
        !$acc end host_data
        !
        !$acc parallel loop present(tmp_r)
        DO ir = 1, dffts_nnr
           tmp_r(ir) = tmp_r(ir) + w1*REAL(psic(ir),KIND=DP)*AIMAG(psic(ir))
        ENDDO
        !$acc end parallel
        !
     ENDDO
     !
     !$acc parallel loop present(drho,tmp_r)
     DO ir = 1, dffts_nnr
        drho(ir,current_spin) = CMPLX(tmp_r(ir),KIND=DP)
     ENDDO
     !$acc end parallel
     !
  ENDDO
  !
  !$acc update host(drho)
  CALL mp_sum(drho,inter_pool_comm)
  CALL mp_sum(drho,inter_bgrp_comm)
  !
#if !defined(__CUDA)
  DEALLOCATE(tmp_r)
#endif
  !
#if defined(__CUDA)
  CALL stop_clock_gpu('calc_dens')
#else
  CALL stop_clock('calc_dens')
#endif
  !
END SUBROUTINE
