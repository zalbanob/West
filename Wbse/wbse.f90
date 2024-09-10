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
!-----------------------------------------------------------------------
PROGRAM wbse
  !-----------------------------------------------------------------------
  !
  ! This is the main program that calculates the static screening.
  !
  USE check_stop,           ONLY : check_stop_init
  USE mp_global,            ONLY : mp_startup,mp_global_end
  USE west_environment,     ONLY : west_environment_start,west_environment_end
  USE westcom,              ONLY : l_davidson,l_lanczos
  use nvtx
  !
  IMPLICIT NONE
  !
  CHARACTER(LEN=9) :: code = 'WBSE'
  !
  ! *** START ***
  !
  CALL check_stop_init( )
  !
  ! Initialize MPI, clocks, print initial messages
  !
#if defined(__MPI)
  CALL mp_startup( start_images = .TRUE. )
#endif
  !
  call nvtxStartRange("west_environment_start")
  CALL west_environment_start( code )
  CALL nvtxEndRange
  !
  CALL nvtxStartRange("west_readin")
  CALL west_readin( code )
  CALL nvtxEndRange
  !
  CALL nvtxStartRange("wbse_setup")
  CALL wbse_setup( )
  CALL nvtxEndRange
  !
  IF( l_davidson ) THEN
     CALL nvtxStartRange("wbse_davidson_diago")
     CALL wbse_davidson_diago( )
     CALL nvtxEndRange
  ENDIF
  !
  IF( l_lanczos ) THEN
     CALL nvtxStartRange("wbse_lanczos_diago")
     CALL wbse_lanczos_diago( )
     CALL nvtxEndRange
  ENDIF
  !
  CALL nvtxStartRange("exx_ungo")
  CALL exx_ungo( )
  CALL nvtxEndRange
  !
  CALL nvtxStartRange("clean_scratchfiles")
  CALL clean_scratchfiles( )
  CALL nvtxEndRange
  !
  CALL west_print_clocks( )
  !
  CALL west_environment_end( code )
  !
  CALL mp_global_end( )
  !
END PROGRAM
