!
! Copyright (C) 2015-2019 M. Govoni
! This file is distributed under the terms of the
! GNU General Public License. See the file License
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
! This file is part of WEST.
!
! Contributors to this file:
! Marco Govoni
!
!-----------------------------------------------------------------------
SUBROUTINE add_intput_parameters_to_json_file( num_drivers, driver, json )
  !-----------------------------------------------------------------------
  !
  USE json_module,      ONLY : json_file
  USE pwcom
  USE westcom
  USE io_files,         ONLY : tmp_dir, prefix
  USE io_global,        ONLY : stdout
  USE mp,               ONLY : mp_bcast
  USE mp_world,         ONLY : mpime,root,world_comm
  USE mp_global,        ONLY : nimage
  USE io_push,          ONLY : io_push_title,io_push_value,io_push_bar,io_push_es0,io_push_c512
  !
  IMPLICIT NONE
  !
  ! I/O
  !
  INTEGER, INTENT(IN) :: num_drivers
  INTEGER, INTENT(IN) :: driver(num_drivers)
  TYPE(json_file), INTENT(INOUT) :: json
  !
  IF ( mpime == root ) THEN
     !
     IF ( ANY(driver(:)==1) ) THEN
        !
        CALL json%add('input.input_west.qe_prefix',TRIM(qe_prefix))
        CALL json%add('input.input_west.west_prefix',TRIM(west_prefix))
        CALL json%add('input.input_west.outdir',TRIM(outdir))
        !
     ENDIF
     !
     IF ( ANY(driver(:)==2) ) THEN
        !
        CALL json%add('input.wstat_control.wstat_calculation',TRIM(wstat_calculation))
        CALL json%add('input.wstat_control.n_pdep_eigen',n_pdep_eigen)
        CALL json%add('input.wstat_control.n_pdep_times',n_pdep_times)
        CALL json%add('input.wstat_control.n_pdep_maxiter',n_pdep_maxiter)
        CALL json%add('input.wstat_control.n_dfpt_maxiter',n_dfpt_maxiter)
        CALL json%add('input.wstat_control.n_pdep_read_from_file',n_pdep_read_from_file)
        CALL json%add('input.wstat_control.trev_pdep',trev_pdep)
        CALL json%add('input.wstat_control.trev_pdep_rel',trev_pdep_rel)
        CALL json%add('input.wstat_control.tr2_dfpt',tr2_dfpt)
        CALL json%add('input.wstat_control.l_kinetic_only',l_kinetic_only)
        CALL json%add('input.wstat_control.l_minimize_exx_if_active',l_minimize_exx_if_active)
        CALL json%add('input.wstat_control.l_use_ecutrho',l_use_ecutrho)
        CALL json%add('input.wstat_control.qlist', qlist)
        !
     ENDIF
     !
     IF ( ANY(driver(:)==3) ) THEN
        !
        CALL json%add('input.wfreq_control.wfreq_calculation',TRIM(wfreq_calculation))
        CALL json%add('input.wfreq_control.n_pdep_eigen_to_use',n_pdep_eigen_to_use)
        CALL json%add('input.wfreq_control.qp_bandrange',qp_bandrange)
        CALL json%add('input.wfreq_control.macropol_calculation',macropol_calculation)
        CALL json%add('input.wfreq_control.n_lanczos',n_lanczos)
        CALL json%add('input.wfreq_control.n_imfreq',n_imfreq)
        CALL json%add('input.wfreq_control.n_refreq',n_refreq)
        CALL json%add('input.wfreq_control.ecut_imfreq',ecut_imfreq)
        CALL json%add('input.wfreq_control.ecut_refreq',ecut_refreq)
        CALL json%add('input.wfreq_control.wfreq_eta',wfreq_eta)
        CALL json%add('input.wfreq_control.n_secant_maxiter',n_secant_maxiter)
        CALL json%add('input.wfreq_control.trev_secant',trev_secant)
        CALL json%add('input.wfreq_control.l_enable_lanczos',l_enable_lanczos)
        CALL json%add('input.wfreq_control.l_enable_gwetot',l_enable_gwetot)
        CALL json%add('input.wfreq_control.o_restart_time',o_restart_time)
        CALL json%add('input.wfreq_control.ecut_spectralf',ecut_spectralf)
        CALL json%add('input.wfreq_control.n_spectralf',n_spectralf)
        !
     ENDIF
     !
     IF ( ANY(driver(:)==4) ) THEN
        !
        CALL json%add('input.westpp_control.westpp_calculation',TRIM(westpp_calculation))
        CALL json%add('input.westpp_control.westpp_range',westpp_range)
        CALL json%add('input.westpp_control.westpp_format',TRIM(westpp_format))
        CALL json%add('input.westpp_control.westpp_sign',westpp_sign)
        CALL json%add('input.westpp_control.westpp_n_pdep_eigen_to_use',westpp_n_pdep_eigen_to_use)
        CALL json%add('input.westpp_control.westpp_r0',westpp_r0)
        CALL json%add('input.westpp_control.westpp_nr',westpp_nr)
        CALL json%add('input.westpp_control.westpp_rmax',westpp_rmax)
        CALL json%add('input.westpp_control.westpp_epsinfty',westpp_epsinfty)
        !
     ENDIF
     !
     IF ( ANY(driver(:)==5) ) THEN
        !
        CALL json%add('input.server_control.document',TRIM(document))
        !
     ENDIF
     !
     IF ( ANY(driver(:)==6)) THEN
        !
        CALL json%add('input.wbse_init_control.wbse_init_calculation',TRIM( wbse_init_calculation ))
        !CALL json%add('input.wbse_init_control.n_pdep_eigen',n_pdep_eigen)
        CALL json%add('input.wbse_init_control.localization',TRIM(localization))
        CALL json%add('input.wbse_init_control.chi_kernel',TRIM( chi_kernel ))
        CALL json%add('input.wbse_init_control.which_spin_channel',which_spin_channel)
        CALL json%add('input.wbse_init_control.overlap_thr',overlap_thr)
        !CALL json%add('input.wbse_init_control.l_use_localise_repr',l_use_localise_repr)
        !CALL json%add('input.wbse_init_control.l_use_bisection_thr',l_use_bisection_thr)
        CALL json%add('input.wbse_init_control.wfc_from_qbox',TRIM(wfc_from_qbox))
        !
     END IF
     !
     IF ( ANY(driver(:)==7)) THEN
        CALL json%add('input.wbse_control.wbse_calculation',TRIM( wbse_calculation ))
        CALL json%add('input.wbse_control.solver',TRIM( solver ))
        CALL json%add('input.wbse_control.qp_correction',TRIM( qp_correction ))
        CALL json%add('input.wbse_control.scissor_ope',scissor_ope)
        CALL json%add('input.wbse_control.n_liouville_eigen',n_liouville_eigen)
        CALL json%add('input.wbse_control.n_liouville_times',n_liouville_times)
        CALL json%add('input.wbse_control.n_liouville_maxiter',n_liouville_maxiter)
        CALL json%add('input.wbse_control.n_liouville_read_from_file',n_liouville_read_from_file)
        CALL json%add('input.wbse_control.trev_plep',trev_liouville)
        CALL json%add('input.wbse_control.trev_plep_rel',trev_liouville_rel)
        CALL json%add('input.wbse_control.n_lanczos',n_lanczos)
        CALL json%add('input.wbse_control.ipol_input', TRIM(ipol_input))
        CALL json%add('input.wbse_control.macropol_calculation', TRIM(wbse_macropol_calculation))
        CALL json%add('input.wbse_control.epsinfty',epsinfty)
        CALL json%add('input.wbse_control.spin_excitation',TRIM(spin_excitation))
        !CALL json%add('input.wbse_control.l_bse_calculation',l_bse_calculation)
        !CALL json%add('input.wbse_control.l_qp_correction',l_qp_correction)
        !CALL json%add('input.wbse_control.l_diag_term_only',l_diag_term_only)
        CALL json%add('input.wbse_control.l_preconditioning',l_preconditioning)
        !CALL json%add('input.wbse_control.wbse_diag_method',TRIM(wbse_diag_method))
        !CALL json%add('input.wbse_control.macropol_dfpt',macropol_dfpt)
     !
     END IF
     !
!     IF ( ANY(driver(:)==8)) THEN
!        !
!        CALL json%add('input.qbox_control.nrowmax',nrowmax)
!        CALL json%add('input.qbox_control.xml_file',TRIM(xml_file))
!        CALL json%add('input.qbox_control.xc',TRIM(xc))
!        CALL json%add('input.qbox_control.alpha_pbe0',alpha_pbe0)
!        CALL json%add('input.qbox_control.amplitude',amplitude)
!        CALL json%add('input.qbox_control.wf_dyn',TRIM(wf_dyn))
!        CALL json%add('input.qbox_control.btHF',btHF)
!        CALL json%add('input.qbox_control.blHF', TRIM(blHF))
!        CALL json%add('input.qbox_control.nitscf',nitscf)
!        CALL json%add('input.qbox_control.nite',nite)
!        !
!     !
!     END IF
     !
     IF (ANY(driver(:)==8)) THEN
        !
        CALL json%add('input.wbsepp_control.wbsepp_type',wbsepp_type)
        CALL json%add('input.wbsepp_control.n_plep_read_from_file',n_liouville_read_from_file)
        CALL json%add('input.wbsepp_control.macropol_dfpt',macropol_dfpt)
        CALL json%add('input.wbsepp_control.r0_input', r0_input)
        CALL json%add('input.wbsepp_control.iexc_plot',iexc_plot)
        CALL json%add('input.wbsepp_control.itermax',itermax)
        CALL json%add('input.wbsepp_control.itermax0',itermax0)
        CALL json%add('input.wbsepp_control.ipol',ipol)
        CALL json%add('input.wbsepp_control.sym_op',sym_op)
        CALL json%add('input.wbsepp_control.units',units)
        CALL json%add('input.wbsepp_control.verbosity',verbosity)
        CALL json%add('input.wbsepp_control.extrapolation',TRIM(extrapolation))
        CALL json%add('input.wbsepp_control.start',start)
        CALL json%add('input.wbsepp_control.end',end)
        CALL json%add('input.wbsepp_control.increment',increment)
        CALL json%add('input.wbsepp_control.epsil',epsil)
        CALL json%add('input.wbsepp_control.spin_channel',spin_channel)
        !
     END IF
     !
  ENDIF
  !
END SUBROUTINE


SUBROUTINE fetch_input_yml( num_drivers, driver, verbose, debug )
  !
  USE west_version, ONLY : start_forpy, end_forpy
  USE io_push,      ONLY : io_push_title,io_push_value,io_push_bar,io_push_es0,io_push_c512
  USE forpy_mod,    ONLY: call_py, call_py_noret, import_py, module_py
  USE forpy_mod,    ONLY: tuple, tuple_create
  USE forpy_mod,    ONLY: dict, dict_create
  USE forpy_mod,    ONLY: list, list_create
  USE forpy_mod,    ONLY: object, cast
  USE forpy_mod,    ONLY : exception_matches, KeyError, err_clear, err_print
  USE westcom
  USE io_files,         ONLY : tmp_dir, prefix
  USE io_global,        ONLY : stdout
  USE mp,               ONLY : mp_bcast, mp_barrier
  USE mp_world,         ONLY : mpime,root,world_comm
  USE mp_global,        ONLY : nimage
  USE gvect,            ONLY : ecutrho
  USE start_k,          ONLY : nk1, nk2, nk3
  USE control_flags,    ONLY : gamma_only
  USE json_module,      ONLY : json_file
  USE pwcom,            ONLY : nelec
  !
  IMPLICIT NONE
  !
  ! I/O
  !
  INTEGER, INTENT(IN) :: num_drivers
  INTEGER, INTENT(IN) :: driver(num_drivers)
  LOGICAL, INTENT(IN) :: verbose
  LOGICAL, INTENT(IN) :: debug
  !
  INTEGER :: IERR
  TYPE(tuple) :: args
  TYPE(dict) :: kwargs
  TYPE(module_py) :: pymod
  TYPE(object) :: return_obj, tmp_obj
  TYPE(dict) :: return_dict
  TYPE(list) :: tmp_list
  INTEGER :: list_len
  INTEGER :: i
  INTEGER :: nq
  INTEGER :: numsp
  CHARACTER(LEN=512), EXTERNAL :: trimcheck
  CHARACTER(LEN=:),ALLOCATABLE :: cvalue
  TYPE(json_file) :: json
  INTEGER :: iunit, lenc
  INTEGER, PARAMETER :: DUMMY_DEFAULT = -1210
  !
  CALL start_clock('fetch_input')
  !
  IF ( mpime==root ) THEN
     !
     IERR = import_py(pymod, "west_fetch_input")
     !
     IF ( ANY(driver(:)==1) ) THEN
        !
        IERR = tuple_create(args, 3)
        IERR = args%setitem(0, TRIM(ADJUSTL(main_input_file)) )
        IERR = args%setitem(1, "input_west" )
        IERR = args%setitem(2, verbose )
        IERR = dict_create(kwargs)
        !
        IERR = call_py(return_obj, pymod, "read_keyword_from_file", args, kwargs)
        IERR = cast(return_dict, return_obj)
        !
        CALL args%destroy
        CALL kwargs%destroy
        CALL return_obj%destroy
        !
        IERR = return_dict%getitem(cvalue, "qe_prefix"); qe_prefix = TRIM(ADJUSTL(cvalue))
        IERR = return_dict%getitem(cvalue, "west_prefix"); west_prefix = TRIM(ADJUSTL(cvalue))
        IERR = return_dict%getitem(cvalue, "outdir"); outdir = TRIM(ADJUSTL(cvalue))
        !
        CALL return_dict%destroy
        !
     ENDIF
     !
     IF ( ANY(driver(:)==2) ) THEN
        !
        IF ( gamma_only ) THEN
           nq = 1
        ELSE
           nq = nk1*nk2*nk3
        ENDIF
        !
        IERR = tuple_create(args, 3)
        IERR = args%setitem(0, TRIM(ADJUSTL(main_input_file)) )
        IERR = args%setitem(1, "wstat_control" )
        IERR = args%setitem(2, verbose )
        IERR = dict_create(kwargs)
        IERR = kwargs%setitem("nq",nq)
        IERR = kwargs%setitem("nelec",nelec)
        !
        IERR = call_py(return_obj, pymod, "read_keyword_from_file", args, kwargs)
        IERR = cast(return_dict, return_obj)
        !
        CALL args%destroy
        CALL kwargs%destroy
        CALL return_obj%destroy
        !
        IERR = return_dict%getitem(cvalue, "wstat_calculation"); wstat_calculation = TRIM(ADJUSTL(cvalue))
        IERR = return_dict%get(n_pdep_eigen, "n_pdep_eigen", DUMMY_DEFAULT)
        IERR = return_dict%get(n_pdep_times, "n_pdep_times", DUMMY_DEFAULT)
        IERR = return_dict%get(n_pdep_maxiter, "n_pdep_maxiter", DUMMY_DEFAULT)
        IERR = return_dict%get(n_dfpt_maxiter, "n_dfpt_maxiter", DUMMY_DEFAULT)
        IERR = return_dict%get(n_pdep_read_from_file, "n_pdep_read_from_file", DUMMY_DEFAULT)
        IERR = return_dict%getitem(trev_pdep, "trev_pdep")
        IERR = return_dict%getitem(trev_pdep_rel, "trev_pdep_rel")
        IERR = return_dict%getitem(tr2_dfpt, "tr2_dfpt")
        IERR = return_dict%getitem(l_kinetic_only, "l_kinetic_only")
        IERR = return_dict%getitem(l_minimize_exx_if_active, "l_minimize_exx_if_active")
        IERR = return_dict%getitem(l_use_ecutrho, "l_use_ecutrho")
        IERR = return_dict%getitem(tmp_obj, "qlist")
        IERR = cast(tmp_list,tmp_obj)
        IERR = tmp_list%len(list_len)
        IF( ALLOCATED(qlist) ) DEALLOCATE(qlist)
        ALLOCATE(qlist(list_len))
        DO i = 0, list_len-1 ! Python indices start at 0
           IERR = tmp_list%getitem(qlist(i+1), i) ! Fortran indices start at 1
        ENDDO
        CALL tmp_list%destroy
        CALL tmp_obj%destroy
        !
        CALL return_dict%destroy
        !
     ENDIF
     !
     IF ( ANY(driver(:)==3) ) THEN
        !
        IERR = tuple_create(args, 3)
        IERR = args%setitem(0, TRIM(ADJUSTL(main_input_file)) )
        IERR = args%setitem(1, "wfreq_control" )
        IERR = args%setitem(2, verbose )
        IERR = dict_create(kwargs)
        IERR = kwargs%setitem("nelec",nelec)
        IERR = kwargs%setitem("ecutrho",ecutrho)
        !
        IERR = call_py(return_obj, pymod, "read_keyword_from_file", args, kwargs)
        IERR = cast(return_dict, return_obj)
        !
        CALL args%destroy
        CALL kwargs%destroy
        CALL return_obj%destroy
        !
        IERR = return_dict%getitem(cvalue, "wfreq_calculation"); wfreq_calculation = TRIM(ADJUSTL(cvalue))
        IERR = return_dict%get(n_pdep_eigen_to_use, "n_pdep_eigen_to_use", DUMMY_DEFAULT)
        IERR = return_dict%getitem(tmp_obj, "qp_bandrange")
        IERR = cast(tmp_list,tmp_obj)
        IERR = tmp_list%len(list_len)
        IERR = tmp_list%getitem(qp_bandrange(1), 0) ! Fortran indices start at 1
        IERR = tmp_list%getitem(qp_bandrange(2), 1) ! Fortran indices start at 1
        CALL tmp_list%destroy
        CALL tmp_obj%destroy
        IERR = return_dict%getitem(cvalue, "macropol_calculation"); macropol_calculation = TRIM(ADJUSTL(cvalue))
        IERR = return_dict%get(n_lanczos, "n_lanczos", DUMMY_DEFAULT)
        IERR = return_dict%get(n_imfreq, "n_imfreq", DUMMY_DEFAULT)
        IERR = return_dict%get(n_refreq, "n_refreq", DUMMY_DEFAULT)
        IERR = return_dict%getitem(ecut_imfreq, "ecut_imfreq")
        IERR = return_dict%getitem(ecut_refreq, "ecut_refreq")
        IERR = return_dict%getitem(wfreq_eta, "wfreq_eta")
        IERR = return_dict%get(n_secant_maxiter, "n_secant_maxiter", DUMMY_DEFAULT)
        IERR = return_dict%getitem(trev_secant, "trev_secant")
        IERR = return_dict%getitem(l_enable_lanczos, "l_enable_lanczos")
        IERR = return_dict%getitem(l_enable_gwetot, "l_enable_gwetot")
        IERR = return_dict%getitem(o_restart_time, "o_restart_time")
        IERR = return_dict%getitem(tmp_obj, "ecut_spectralf")
        IERR = cast(tmp_list,tmp_obj)
        IERR = tmp_list%getitem(ecut_spectralf(1), 0) ! Fortran indices start at 1 
        IERR = tmp_list%getitem(ecut_spectralf(2), 1) ! Fortran indices start at 1 
        CALL tmp_list%destroy 
        CALL tmp_obj%destroy 
        IERR = return_dict%get(n_spectralf, "n_spectralf", DUMMY_DEFAULT)
        !
        CALL return_dict%destroy
        !
     ENDIF
     !
     IF ( ANY(driver(:)==4) ) THEN
        !
        IERR = tuple_create(args, 3)
        IERR = args%setitem(0, TRIM(ADJUSTL(main_input_file)) )
        IERR = args%setitem(1, "westpp_control" )
        IERR = args%setitem(2, verbose )
        IERR = dict_create(kwargs)
        !
        IERR = call_py(return_obj, pymod, "read_keyword_from_file", args, kwargs)
        IERR = cast(return_dict, return_obj)
        !
        CALL args%destroy
        CALL kwargs%destroy
        CALL return_obj%destroy
        !
        IERR = return_dict%getitem(cvalue, "westpp_calculation"); westpp_calculation = TRIM(ADJUSTL(cvalue))
        IERR = return_dict%getitem(tmp_obj, "westpp_range")
        IERR = cast(tmp_list,tmp_obj)
        IERR = tmp_list%getitem(westpp_range(1), 0) ! Fortran indices start at 1
        IERR = tmp_list%getitem(westpp_range(2), 1) ! Fortran indices start at 1
        CALL tmp_list%destroy
        CALL tmp_obj%destroy
        IERR = return_dict%getitem(cvalue, "westpp_format"); westpp_format = TRIM(ADJUSTL(cvalue))
        IERR = return_dict%getitem(westpp_sign, "westpp_sign")
        IERR = return_dict%get(westpp_n_pdep_eigen_to_use, "westpp_n_pdep_eigen_to_use", DUMMY_DEFAULT)
        IERR = return_dict%getitem(tmp_obj, "westpp_r0")
        IERR = cast(tmp_list,tmp_obj)
        IERR = tmp_list%getitem(westpp_r0(1), 0) ! Fortran indices start at 1 
        IERR = tmp_list%getitem(westpp_r0(2), 1) ! Fortran indices start at 1 
        IERR = tmp_list%getitem(westpp_r0(3), 2) ! Fortran indices start at 1 
        CALL tmp_list%destroy 
        CALL tmp_obj%destroy 
        IERR = return_dict%get(westpp_nr, "westpp_nr", DUMMY_DEFAULT)
        IERR = return_dict%getitem(westpp_rmax, "westpp_rmax")
        IERR = return_dict%getitem(westpp_epsinfty, "westpp_epsinfty")
        !
        CALL return_dict%destroy
        !
     ENDIF
     !
     IF ( ANY(driver(:)==5) ) THEN
        !
        IERR = tuple_create(args, 3)
        IERR = args%setitem(0, TRIM(ADJUSTL(main_input_file)) )
        IERR = args%setitem(1, "server_control" )
        IERR = args%setitem(2, verbose )
        IERR = dict_create(kwargs)
        !
        IERR = call_py(return_obj, pymod, "read_keyword_from_file", args, kwargs)
        IERR = cast(return_dict, return_obj)
        !
        CALL args%destroy
        CALL kwargs%destroy
        CALL return_obj%destroy
        !
        IERR = return_dict%getitem(cvalue, "document"); document = TRIM(ADJUSTL(cvalue))
        !
        CALL return_dict%destroy
        !
     ENDIF
     !
     IF ( ANY(driver(:)==6) ) THEN
        !
        !wbse_init gamma only qlist = [1]
        IF( ALLOCATED(qlist) ) DEALLOCATE(qlist)
        ALLOCATE(qlist(1))
        qlist = (/1/)
        !
        IERR = tuple_create(args, 3)
        IERR = args%setitem(0, TRIM(ADJUSTL(main_input_file)) )
        IERR = args%setitem(1, "wbse_init_control" )
        IERR = args%setitem(2, verbose )
        IERR = dict_create(kwargs)
        !
        IERR = call_py(return_obj, pymod, "read_keyword_from_file", args, kwargs)
        IERR = cast(return_dict, return_obj)
        !
        CALL args%destroy
        CALL kwargs%destroy
        CALL return_obj%destroy
        !
        IERR = return_dict%getitem(cvalue, "wbse_init_calculation"); wbse_init_calculation = TRIM(&
               ADJUSTL(cvalue))
        IERR = return_dict%getitem(cvalue, "localization"); localization = TRIM(ADJUSTL(cvalue))
        IERR = return_dict%getitem(cvalue, "bisection_info"); bisection_info = TRIM(ADJUSTL(cvalue))
        !IERR = return_dict%getitem(cvalue, "which_bse_method"); which_bse_method = TRIM(&
        !       ADJUSTL(cvalue))
        !IERR = return_dict%getitem(n_pdep_eigen, "n_pdep_eigen")
        IERR = return_dict%getitem(cvalue, "chi_kernel"); chi_kernel = TRIM(ADJUSTL(cvalue))
        IERR = return_dict%getitem(which_spin_channel, "which_spin_channel")
        IERR = return_dict%getitem(overlap_thr, "overlap_thr")
        !IERR = return_dict%getitem(l_use_localise_repr, "l_use_localise_repr")
        !IERR = return_dict%getitem(l_use_bisection_thr, "l_use_bisection_thr")
        IERR = return_dict%getitem(cvalue, "wfc_from_qbox"); wfc_from_qbox = TRIM(&
               ADJUSTL(cvalue))
        !
        CALL return_dict%destroy
        !
     ENDIF
     !
     IF ( ANY(driver(:)==7) ) THEN
        IERR = tuple_create(args, 3)
        IERR = args%setitem(0, TRIM(ADJUSTL(main_input_file)) )
        IERR = args%setitem(1, "wbse_control" )
        IERR = args%setitem(2, verbose )
        IERR = dict_create(kwargs)
        !
        IERR = call_py(return_obj, pymod, "read_keyword_from_file", args, kwargs)
        IERR = cast(return_dict, return_obj)
        !
        CALL args%destroy
        CALL kwargs%destroy
        CALL return_obj%destroy
        !
        IERR = return_dict%getitem(cvalue, "wbse_calculation");wbse_calculation = TRIM(ADJUSTL(cvalue))
        IERR = return_dict%getitem(cvalue, "solver");solver = TRIM(ADJUSTL(cvalue))
        IERR = return_dict%getitem(cvalue, "qp_correction"); qp_correction = TRIM(ADJUSTL(cvalue))
        IERR = return_dict%getitem(scissor_ope, "scissor_ope")
        IERR = return_dict%getitem(n_liouville_eigen, "n_liouville_eigen")
        IERR = return_dict%getitem(n_liouville_times, "n_liouville_times")
        IERR = return_dict%getitem(n_liouville_maxiter, "n_liouville_maxiter")
        IERR = return_dict%getitem(n_liouville_read_from_file, "n_liouville_read_from_file")
        IERR = return_dict%getitem(trev_liouville, "trev_liouville")
        IERR = return_dict%getitem(trev_liouville_rel, "trev_liouville_rel")
        IERR = return_dict%getitem(n_lanczos,"n_lanczos")
        !IERR = return_dict%getitem(n_lzstep, "n_lzstep")
        IERR = return_dict%getitem(cvalue, "ipol_input");ipol_input = TRIM(ADJUSTL(cvalue))
        IERR = return_dict%getitem(cvalue, "macropol_calculation"); wbse_macropol_calculation =  TRIM(ADJUSTL(cvalue))
        !IERR = return_dict%getitem(eps_macro, "eps_macro")
        IERR = return_dict%getitem(epsinfty,"epsinfty")
        IERR = return_dict%getitem(cvalue, "spin_excitation");spin_excitation = TRIM(ADJUSTL(cvalue))
        IERR = return_dict%getitem(l_preconditioning, "l_preconditioning")
        !
        !IERR = return_dict%getitem(l_bse_calculation, "l_bse_calculation")
        !IERR = return_dict%getitem(l_qp_correction, "l_qp_correction")
        !IERR = return_dict%getitem(l_diag_term_only, "l_diag_term_only")
        !IERR = return_dict%getitem(cvalue, "wbse_diag_method");wbse_diag_method = TRIM(ADJUSTL(cvalue))
        !IERR = return_dict%getitem(macropol_dfpt, "macropol_dfpt")
        !
        CALL return_dict%destroy
        !
     ENDIF
     !
!     IF ( ANY(driver(:)==8) ) THEN
!        IERR = tuple_create(args, 3)
!        IERR = args%setitem(0, TRIM(ADJUSTL(main_input_file)) )
!        IERR = args%setitem(1, "qbox_control" )
!        IERR = args%setitem(2, verbose )
!        IERR = dict_create(kwargs)
!        !
!        IERR = call_py(return_obj, pymod, "read_keyword_from_file", args, kwargs)
!        IERR = cast(return_dict, return_obj)
!        !
!        CALL args%destroy
!        CALL kwargs%destroy
!        CALL return_obj%destroy
!        !
!        IERR = return_dict%getitem(nrowmax, "nrowmax")
!        IERR = return_dict%getitem(cvalue, "xml_file");xml_file = TRIM(ADJUSTL(cvalue))
!        IERR = return_dict%getitem(cvalue, "xc");xc = TRIM(ADJUSTL(cvalue))
!        IERR = return_dict%getitem(alpha_pbe0, "alpha_pbe0")
!        IERR = return_dict%getitem(amplitude, "amplitude")
!        IERR = return_dict%getitem(cvalue, "wf_dyn");wf_dyn = TRIM(ADJUSTL(cvalue))
!        IERR = return_dict%getitem(btHF, "btHF")
!        IERR = return_dict%getitem(cvalue, "blHF");blHF = TRIM(ADJUSTL(cvalue))
!        IERR = return_dict%getitem(nitscf, "nitscf")
!        IERR = return_dict%getitem(nite, "nite")
!        !
!        CALL return_dict%destroy
!        !
!     ENDIF
     !
     IF ( ANY(driver(:)==8) ) THEN
        !wbsepp gamma only qlist = [1]
        IF( ALLOCATED(qlist) ) DEALLOCATE(qlist)
        ALLOCATE(qlist(1))
        qlist = (/1/)
        !
        IERR = tuple_create(args, 3)
        IERR = args%setitem(0, TRIM(ADJUSTL(main_input_file)) )
        IERR = args%setitem(1, "wbsepp_control" )
        IERR = args%setitem(2, verbose )
        IERR = dict_create(kwargs)
        !
        IERR = call_py(return_obj, pymod, "read_keyword_from_file", args, kwargs)
        IERR = cast(return_dict, return_obj)
        !
        CALL args%destroy
        CALL kwargs%destroy
        CALL return_obj%destroy
        !
        IERR = return_dict%getitem(wbsepp_type, "wbsepp_type")
        IERR = return_dict%getitem(n_liouville_read_from_file, "n_liouville_read_from_file")
        IERR = return_dict%getitem(macropol_dfpt, "macropol_dfpt")
        IERR = return_dict%getitem(tmp_obj, "r0_input")
        IERR = cast(tmp_list,tmp_obj)
        IERR = tmp_list%len(list_len)
        IERR = tmp_list%getitem(r0_input(1), 0) ! Fortran indices start at 1
        IERR = tmp_list%getitem(r0_input(2), 1) ! Fortran indices start at 1
        IERR = tmp_list%getitem(r0_input(3), 2) ! Fortran indices start at 1
        CALL tmp_list%destroy
        CALL tmp_obj%destroy
        IERR = return_dict%getitem(iexc_plot, "iexc_plot")
        IERR = return_dict%getitem(itermax, "itermax")
        IERR = return_dict%getitem(itermax0, "itermax0")
        IERR = return_dict%getitem(ipol, "ipol")
        IERR = return_dict%getitem(sym_op, "sym_op")
        IERR = return_dict%getitem(units, "units")
        IERR = return_dict%getitem(verbosity, "verbosity")
        IERR = return_dict%getitem(cvalue, "extrapolation");extrapolation = TRIM(ADJUSTL(cvalue))
        IERR = return_dict%getitem(start, "start")
        IERR = return_dict%getitem(end, "end")
        IERR = return_dict%getitem(increment, "increment")
        IERR = return_dict%getitem(epsil, "epsil")
        IERR = return_dict%getitem(spin_channel, "spin_channel")
        !
        CALL return_dict%destroy
        !
     ENDIF
     !
     CALL pymod%destroy
     !
     !
  ENDIF
  !
  ! BCAST & CHECKS
  !
  IF ( ANY(driver(:)==1) ) THEN
     !
     CALL mp_bcast(qe_prefix,root,world_comm); prefix=qe_prefix
     CALL mp_bcast(west_prefix,root,world_comm)
     CALL mp_bcast(outdir,root,world_comm); tmp_dir = trimcheck (outdir)
     !
  ENDIF
  !
  IF ( ANY(driver(:)==2) ) THEN
     !
     CALL mp_bcast(wstat_calculation,root,world_comm)
     CALL mp_bcast(n_pdep_eigen,root,world_comm)
     CALL mp_bcast(n_pdep_times,root,world_comm)
     CALL mp_bcast(n_pdep_maxiter,root,world_comm)
     CALL mp_bcast(n_dfpt_maxiter,root,world_comm)
     CALL mp_bcast(n_pdep_read_from_file,root,world_comm)
     CALL mp_bcast(trev_pdep,root,world_comm)
     CALL mp_bcast(trev_pdep_rel,root,world_comm)
     CALL mp_bcast(tr2_dfpt,root,world_comm)
     CALL mp_bcast(l_kinetic_only,root,world_comm)
     CALL mp_bcast(l_minimize_exx_if_active,root,world_comm)
     CALL mp_bcast(l_use_ecutrho,root,world_comm)
     IF(mpime == root) nq = SIZE(qlist)
     CALL mp_bcast(nq,root,world_comm)
     IF(mpime /= root) THEN
        IF( ALLOCATED(qlist) ) DEALLOCATE(qlist)
        ALLOCATE(qlist(nq))
     ENDIF
     CALL mp_bcast(qlist,root,world_comm)
     !
     ! CHECKS
     !
     IF( n_pdep_times < 2 ) CALL errore('fetch_input','Err: n_pdep_times<2',1)
     IF( n_pdep_eigen < 1 ) CALL errore('fetch_input','Err: n_pdep_eigen<1',1)
     IF( n_pdep_eigen*n_pdep_times < nimage ) CALL errore('fetch_input','Err: n_pdep_eigen*n_pdep_times<nimage',1)
     IF( n_pdep_maxiter < 1 ) CALL errore('fetch_input','Err: n_pdep_maxiter<1',1)
     IF( n_dfpt_maxiter < 1 ) CALL errore('fetch_input','Err: n_dfpt_maxiter<1',1)
     IF( n_pdep_read_from_file < 0 ) CALL errore('fetch_input','Err: n_pdep_read_from_file<0',1)
     IF( n_pdep_read_from_file > n_pdep_eigen ) CALL errore('fetch_input','Err: n_pdep_read_from_file>n_pdep_eigen',1)
     IF(tr2_dfpt<=0._DP) CALL errore('fetch_input','Err: tr2_dfpt<0.',1)
     IF(trev_pdep<=0._DP) CALL errore('fetch_input','Err: trev_pdep<0.',1)
     IF(trev_pdep_rel<=0._DP) CALL errore('fetch_input','Err: trev_pdep_rel<0.',1)
     IF( n_pdep_eigen == DUMMY_DEFAULT ) CALL errore('fetch_input','Err: cannot read n_pdep_eigen')
     IF( n_pdep_times == DUMMY_DEFAULT ) CALL errore('fetch_input','Err: cannot read n_pdep_times')
     IF( n_pdep_maxiter == DUMMY_DEFAULT ) CALL errore('fetch_input','Err: cannot read n_pdep_maxiter')
     IF( n_dfpt_maxiter == DUMMY_DEFAULT ) CALL errore('fetch_input','Err: cannot read n_dfpt_maxiter')
     IF( n_pdep_read_from_file == DUMMY_DEFAULT ) CALL errore('fetch_input','Err: cannot read n_pdep_read_from_file')
     IF(gamma_only) THEN
        IF (SIZE(qlist)/=1) CALL errore('fetch_input','Err: SIZE(qlist)/=1.',1)
     ELSE
        IF (SIZE(qlist)>nk1*nk2*nk3) CALL errore('fetch_input','Err: SIZE(qlist)>nk1*nk2*nk3.',1)
     ENDIF
     !
  ENDIF
  !
  IF ( ANY(driver(:)==3) ) THEN
     !
     CALL mp_bcast(wfreq_calculation,root,world_comm)
     CALL mp_bcast(n_pdep_eigen_to_use,root,world_comm)
     CALL mp_bcast(qp_bandrange,root,world_comm)
     CALL mp_bcast(macropol_calculation,root,world_comm)
     CALL mp_bcast(n_lanczos,root,world_comm)
     CALL mp_bcast(n_imfreq,root,world_comm)
     CALL mp_bcast(n_refreq,root,world_comm)
     CALL mp_bcast(ecut_imfreq,root,world_comm)
     CALL mp_bcast(ecut_refreq,root,world_comm)
     CALL mp_bcast(wfreq_eta,root,world_comm)
     CALL mp_bcast(n_secant_maxiter,root,world_comm)
     CALL mp_bcast(trev_secant,root,world_comm)
     CALL mp_bcast(l_enable_lanczos,root,world_comm)
     CALL mp_bcast(l_enable_gwetot,root,world_comm)
     CALL mp_bcast(o_restart_time,root,world_comm)
     CALL mp_bcast(ecut_spectralf,root,world_comm)
     CALL mp_bcast(n_spectralf,root,world_comm)
     !
     ! CHECKS
     !
     IF( n_lanczos < 2 ) CALL errore('fetch_input','Err: n_lanczos<2',1)
     IF( n_pdep_eigen_to_use < 1 ) CALL errore('fetch_input','Err: n_pdep_eigen_to_use<1',1)
     IF( n_pdep_eigen_to_use > n_pdep_eigen ) CALL errore('fetch_input','Err: n_pdep_eigen_to_use>n_pdep_eigen',1)
     IF( n_imfreq < 1 ) CALL errore('fetch_input','Err: n_imfreq<1',1)
     IF( n_refreq < 1 ) CALL errore('fetch_input','Err: n_refreq<1',1)
     IF( n_spectralf < 2 ) CALL errore('fetch_input','Err: n_spectralf<1',1)
     IF( qp_bandrange(1) < 1 ) CALL errore('fetch_input','Err: qp_bandrange(1)<1',1)
     IF( qp_bandrange(2) < 1 ) CALL errore('fetch_input','Err: qp_bandrange(2)<1',1)
     IF( qp_bandrange(2) < qp_bandrange(1) ) CALL errore('fetch_input','Err: qp_bandrange(2)<qp_bandrange(1)',1)
     IF( ecut_imfreq<=0._DP) CALL errore('fetch_input','Err: ecut_imfreq<0.',1)
     IF( ecut_refreq<=0._DP) CALL errore('fetch_input','Err: ecut_imfreq<0.',1)
     IF( ecut_spectralf(2)<ecut_spectralf(1)) CALL errore('fetch_input','Err: ecut_spectralf(2)<ecut_spectralf(1)',1)
     IF( wfreq_eta<=0._DP) CALL errore('fetch_input','Err: wfreq_eta<0.',1)
     IF( n_secant_maxiter < 0 ) CALL errore('fetch_input','Err: n_secant_maxiter<0',1)
     IF( trev_secant<=0._DP) CALL errore('fetch_input','Err: trev_secant<0.',1)
     IF( n_pdep_eigen_to_use == DUMMY_DEFAULT ) CALL errore('fetch_input','Err: cannot read n_pdep_eigen_to_use')
     IF( n_lanczos == DUMMY_DEFAULT ) CALL errore('fetch_input','Err: cannot read n_lanczos')
     IF( n_imfreq == DUMMY_DEFAULT ) CALL errore('fetch_input','Err: cannot read n_imfreq')
     IF( n_refreq == DUMMY_DEFAULT ) CALL errore('fetch_input','Err: cannot read n_refreq')
     IF( n_secant_maxiter == DUMMY_DEFAULT ) CALL errore('fetch_input','Err: cannot read n_secant_maxiter')
     IF( n_spectralf == DUMMY_DEFAULT ) CALL errore('fetch_input','Err: cannot read n_spectralf')
     SELECT CASE(macropol_calculation) 
     CASE('N','n','C','c')
     CASE DEFAULT
        CALL errore('fetch_input','Err: macropol_calculation /= (N,C)',1)
     END SELECT
     !
  ENDIF
  !
  IF ( ANY(driver(:)==4) ) THEN
     !
     CALL mp_bcast(westpp_calculation,root,world_comm)
     CALL mp_bcast(westpp_range,root,world_comm)
     CALL mp_bcast(westpp_format,root,world_comm)
     CALL mp_bcast(westpp_sign,root,world_comm)
     CALL mp_bcast(westpp_n_pdep_eigen_to_use,root,world_comm)
     CALL mp_bcast(westpp_r0,root,world_comm)
     CALL mp_bcast(westpp_nr,root,world_comm)
     CALL mp_bcast(westpp_rmax,root,world_comm)
     CALL mp_bcast(westpp_epsinfty,root,world_comm)
     !
     ! CHECKS
     !
     IF( westpp_range(1) < 1 ) CALL errore('fetch_input','Err: westpp_range(1)<1',1)
     IF( westpp_range(2) < 1 ) CALL errore('fetch_input','Err: westpp_range(2)<1',1)
     IF( westpp_range(2) < westpp_range(1) ) CALL errore('fetch_input','Err: westpp_range(2)<westpp_range(1)',1)
     IF( westpp_nr < 1 ) CALL errore('fetch_input','Err: westpp_nr<1',1)
     IF( westpp_n_pdep_eigen_to_use < 1 ) CALL errore('fetch_input','Err: westpp_n_pdep_eigen_to_use<1',1)
     IF( westpp_rmax < 0._DP ) CALL errore('fetch_input','Err: westpp_rmax<0.',1)
     IF( westpp_epsinfty < 1._DP ) CALL errore('fetch_input','Err: westpp_epsinfty<1.',1)
     IF( westpp_n_pdep_eigen_to_use == DUMMY_DEFAULT ) CALL errore('fetch_input','Err: cannot read westpp_n_pdep_eigen_to_use')
     IF( westpp_nr == DUMMY_DEFAULT ) CALL errore('fetch_input','Err: cannot read westpp_nr')
     !
  ENDIF
  !
  IF ( ANY(driver(:)==5) ) THEN
     !
     lenc = LEN(document)
     CALL mp_bcast(lenc,root,world_comm)
     IF(mpime/=root) ALLOCATE(CHARACTER(LEN=lenc) :: document)
     CALL mp_bcast(document,root,world_comm)
     !
  ENDIF
  !
  IF ( ANY(driver(:)==6) ) THEN
     !GAMMA ONLY qlist init for wbse_init
     IF(mpime == root) nq = SIZE(qlist)
     CALL mp_bcast(nq,root,world_comm)
     IF(mpime /= root) THEN
        IF( ALLOCATED(qlist) ) DEALLOCATE(qlist)
        ALLOCATE(qlist(nq))
     ENDIF
     CALL mp_bcast(qlist,root,world_comm)
     !
     CALL mp_bcast(wbse_init_calculation,root,world_comm)
     !CALL mp_bcast(which_bse_method,root,world_comm)
     !CALL mp_bcast(l_use_bisection_thr,root,world_comm)
     CALL mp_bcast(overlap_thr,root,world_comm)
     CALL mp_bcast(chi_kernel,root,world_comm)
     !CALL mp_bcast(n_pdep_eigen,root,world_comm)
     CALL mp_bcast(which_spin_channel,root,world_comm)
     !CALL mp_bcast(l_use_localise_repr,root,world_comm)
     CALL mp_bcast(localization,root,world_comm)
     CALL mp_bcast(wfc_from_qbox,root,world_comm)
     CALL mp_bcast(bisection_info,root,world_comm)

     SELECT CASE(TRIM(localization))
     CASE('N','n')
     l_use_localise_repr = .FALSE.
     l_use_bisection_thr = .FALSE.
     CASE('B','b')
     l_use_localise_repr = .TRUE.
     l_use_bisection_thr = .TRUE.
     CASE DEFAULT
        CALL errore('fetch_input','Err: localization != N/B', 1)
     END SELECT
     CALL mp_bcast(l_use_localise_repr,root,world_comm)
     CALL mp_bcast(l_use_bisection_thr,root,world_comm)
     !
     !use_qbox = .FALSE.
     !use_wstat_pdep = .FALSE.
     !SELECT CASE(TRIM(which_bse_method))
     !CASE('finite-field','FF')
     !use_qbox = .TRUE.
     !CASE('PDEP','pdep')
     !use_wstat_pdep = .TRUE.
     !CASE DEFAULT
     !   CALL errore('fetch_input','Err: which_bse_method != FF/PDEP', 1)
     !END SELECT
     !l_test_ovl = .FALSE.
     !IF (wbse_init_calculation == 'I') l_test_ovl = .TRUE.
     !
     !CALL mp_bcast(use_qbox,root,world_comm)
     !CALL mp_bcast(use_wstat_pdep,root,world_comm)
     !CALL mp_bcast(l_test_ovl,root,world_comm)
     !
  ENDIF
  !
  IF ( ANY(driver(:)==7) ) THEN
     !
     CALL mp_bcast(wbse_calculation,root,world_comm)
     CALL mp_bcast(solver,root,world_comm)
     CALL mp_bcast(qp_correction,root,world_comm)
     CALL mp_bcast(scissor_ope,root,world_comm)
     CALL mp_bcast(n_liouville_eigen,root,world_comm)
     CALL mp_bcast(n_liouville_times,root,world_comm)
     CALL mp_bcast(n_liouville_maxiter,root,world_comm)
     CALL mp_bcast(n_liouville_read_from_file,root,world_comm)
     CALL mp_bcast(trev_liouville,root,world_comm)
     CALL mp_bcast(trev_liouville_rel,root,world_comm)
     CALL mp_bcast(n_lanczos,root,world_comm)
     CALL mp_bcast(ipol_input,root,world_comm)
     CALL mp_bcast(wbse_macropol_calculation,root,world_comm)
     CALL mp_bcast(epsinfty,root,world_comm)
     CALL mp_bcast(spin_excitation,root,world_comm)
     CALL mp_bcast(l_preconditioning,root,world_comm)
     !CALL mp_bcast(l_bse_calculation,root,world_comm)
     !CALL mp_bcast(l_diag_term_only,root,world_comm)
     !CALL mp_bcast(wbse_diag_method,root,world_comm)
     !CALL mp_bcast(l_qp_correction,root,world_comm)
     !CALL mp_bcast(n_lzstep,root,world_comm)
     !CALL mp_bcast(macropol_dfpt,root,world_comm)
     !
     ! CHECKS
     !
     !TODO: QUESTION, macropol_dfpt depend on wbse_macropol_calculation?
     SELECT CASE(wbse_macropol_calculation)
     CASE('N','n')
        macropol_dfpt = .FALSE.
     CASE('C','c')
        macropol_dfpt = .TRUE.
     CASE DEFAULT
        CALL errore('fetch_input','Err: wbse_macropol_calculation /= N or C',1)
     END SELECT
     !
     SELECT CASE(TRIM(solver))
     CASE('BSE','bse')
        l_bse_calculation = .TRUE.
     CASE('TDDFT','tddft')
        l_bse_calculation = .FALSE.
     CASE DEFAULT
        CALL errore('fetch_input','Err: solver /= BSE or TDDFT',1)
     END SELECT


     SELECT CASE(wbse_calculation)
     CASE('D','d')
        l_davidson = .TRUE.
     CASE('L','l')
        l_lanzcos  = .TRUE.
     CASE DEFAULT
        CALL errore('fetch_input','Err: wbse_calculation /= D d L l',1)
     END SELECT

     IF ( TRIM(qp_correction) == "" ) THEN
        l_qp_correction = .FALSE.
     ELSE
        l_qp_correction = .TRUE.
     ENDIF

     !
     !wlz_calculation = wbse_calculation
     !
     SELECT CASE(spin_excitation)
     CASE('s', 'S', 'singlet')
         l_bse_triplet = .false.
     CASE('t', 'T', 'triplet')
         l_bse_triplet = .true.
     CASE DEFAULT
        CALL errore('fetch_input','Err: spin_excitation /= s or t',1)
     END SELECT
     !
     IF (l_davidson) THEN
        IF( n_liouville_times < 2 ) CALL errore('fetch_input','Err: n_liouville_times<2',1)
        IF( n_liouville_eigen < 1 ) CALL errore('fetch_input','Err: n_liouville_eigen<1',1)
        IF( n_liouville_eigen*n_liouville_times < nimage ) THEN
           CALL errore('fetch_input','Err: n_liouville_eigen*n_liouville_times<nimage',1)
        ENDIF
        IF( n_liouville_maxiter < 1 ) CALL errore('fetch_input','Err: n_liouville_maxiter<1',1)
        IF( n_liouville_read_from_file < 0 ) CALL errore('fetch_input','Err: n_liouville_read_from_file<0',1)
        IF( n_liouville_read_from_file > n_liouville_eigen ) THEN
            CALL errore('fetch_input','Err: n_liouville_read_from_file>n_liouville_eigen',1)
        ENDIF
     ENDIF
     !
     !CALL mp_bcast(wlz_calculation,root,world_comm)
     CALL mp_bcast(l_davidson,root,world_comm)
     CALL mp_bcast(l_lanzcos,root,world_comm)
     CALL mp_bcast(l_bse_triplet,root,world_comm)
     CALL mp_bcast(l_qp_correction,root,world_comm)
     CALL mp_bcast(l_bse_calculation,root,world_comm)
     CALL mp_bcast(macropol_dfpt,root,world_comm)
     !
  ENDIF
  !
!  IF ( ANY(driver(:)==8) ) THEN
!     !
!     CALL mp_bcast(nrowmax,root,world_comm)
!     CALL mp_bcast(xml_file,root,world_comm)
!     CALL mp_bcast(xc,root,world_comm)
!     CALL mp_bcast(amplitude,root,world_comm)
!     CALL mp_bcast(alpha_pbe0,root,world_comm)
!     CALL mp_bcast(wf_dyn,root,world_comm)
!     CALL mp_bcast(blHF,root,world_comm)
!     CALL mp_bcast(btHF,root,world_comm)
!     CALL mp_bcast(nitscf,root,world_comm)
!     CALL mp_bcast(nite,root,world_comm)
!     !
!  ENDIF
  !
  IF ( ANY(driver(:)==8) ) THEN
     !GAMMA ONLY qlist init for wbsepp
     IF(mpime == root) nq = SIZE(qlist)
     CALL mp_bcast(nq,root,world_comm)
     IF(mpime /= root) THEN
        IF( ALLOCATED(qlist) ) DEALLOCATE(qlist)
        ALLOCATE(qlist(nq))
     ENDIF
     CALL mp_bcast(qlist,root,world_comm)
     !
     CALL mp_bcast(wbsepp_type,root,world_comm)
     CALL mp_bcast(n_liouville_read_from_file,root,world_comm)
     CALL mp_bcast(macropol_dfpt,root,world_comm)
     CALL mp_bcast(r0_input,root,world_comm)
     CALL mp_bcast(iexc_plot,root,world_comm)
     CALL mp_bcast(itermax,root,world_comm)
     CALL mp_bcast(itermax0,root,world_comm)
     CALL mp_bcast(ipol,root,world_comm)
     CALL mp_bcast(sym_op,root,world_comm)
     CALL mp_bcast(units,root,world_comm)
     CALL mp_bcast(verbosity,root,world_comm)
     CALL mp_bcast(extrapolation,root,world_comm)
     CALL mp_bcast(start,root,world_comm)
     CALL mp_bcast(end,root,world_comm)
     CALL mp_bcast(increment,root,world_comm)
     CALL mp_bcast(epsil,root,world_comm)
     CALL mp_bcast(spin_channel,root,world_comm)
     !
     !
     l_meg        = .FALSE.
     l_eig_decomp = .FALSE.
     l_lz_spec    = .FALSE.
     l_exc_plot   = .FALSE.
     l_exc_rho_res_plot = .FALSE.
     !
     SELECT CASE(wbsepp_type)
     CASE( 1 )
         l_eig_decomp = .TRUE.
     CASE( 2 )
         l_lz_spec  = .TRUE.
     CASE( 3 )
         l_meg      = .TRUE.
     CASE( 4 )
         l_exc_plot = .TRUE.
     CASE( 5 )
         l_exc_rho_res_plot = .TRUE.
     CASE DEFAULT
        CALL errore('fetch_input','Err: wbsepp_type /= 1,2,3', wbsepp_type)
     END SELECT
     !
     IF ((spin_channel < 1) .OR. (spin_channel > 2)) THEN
        CALL errore('fetch_input','Err: spin_channel/= 1,2', spin_channel)
     ENDIF
     !
     CALL mp_bcast(l_meg,root,world_comm)
     CALL mp_bcast(l_eig_decomp,root,world_comm)
     CALL mp_bcast(l_lz_spec,root,world_comm)
     CALL mp_bcast(l_exc_plot,root,world_comm)
     CALL mp_bcast(l_exc_rho_res_plot,root,world_comm)
     !
  ENDIF

  !
  CALL mp_barrier(world_comm)
  !
  ! REPORT
  !
  IF ( debug .AND. mpime == root ) THEN
     !
     IF ( ANY(driver(:)==1) ) THEN
        !
        ! REPORT
        !
        CALL io_push_title("I/O Summary : input_west")
        !
        numsp = 14
        CALL io_push_c512('qe_prefix',qe_prefix,numsp)
        CALL io_push_c512('west_prefix',west_prefix,numsp)
        CALL io_push_c512('outdir',outdir,numsp)
        !
        CALL io_push_bar()
        !
     ENDIF
     !
     IF ( ANY(driver(:)==2) ) THEN
        !
        ! REPORT
        !
        CALL io_push_title('I/O Summary : wstat_control')
        !
        numsp=30
        CALL io_push_value('wstat_calculation',wstat_calculation,numsp)
        CALL io_push_value('n_pdep_eigen',n_pdep_eigen,numsp)
        CALL io_push_value('n_pdep_times',n_pdep_times,numsp)
        CALL io_push_value('n_pdep_maxiter',n_pdep_maxiter,numsp)
        CALL io_push_value('n_dfpt_maxiter',n_dfpt_maxiter,numsp)
        CALL io_push_value('n_pdep_read_from_file',n_pdep_read_from_file,numsp)
        CALL io_push_es0('trev_pdep',trev_pdep,numsp)
        CALL io_push_es0('trev_pdep_rel',trev_pdep_rel,numsp)
        CALL io_push_es0('tr2_dfpt',tr2_dfpt,numsp)
        CALL io_push_value('l_kinetic_only',l_kinetic_only,numsp)
        CALL io_push_value('l_minimize_exx_if_active',l_minimize_exx_if_active,numsp)
        CALL io_push_value('l_use_ecutrho',l_use_ecutrho,numsp)
        DO i = 1, SIZE(qlist)
           CALL io_push_value('qlist',qlist(i),numsp)
        ENDDO
        !
        CALL io_push_bar()
        !
     ENDIF
     !
     IF ( ANY(driver(:)==3) ) THEN
        !
        ! REPORT
        !
        CALL io_push_title('I/O Summary : wfreq_control')
        !
        numsp=40
        CALL io_push_value('wfreq_calculation',wfreq_calculation,numsp)
        CALL io_push_value('n_pdep_eigen_to_use',n_pdep_eigen_to_use,numsp)
        CALL io_push_value('qp_bandrange(1)',qp_bandrange(1),numsp)
        CALL io_push_value('qp_bandrange(2)',qp_bandrange(2),numsp)
        CALL io_push_value('macropol_calculation',macropol_calculation,numsp)
        CALL io_push_value('n_lanczos',n_lanczos,numsp)
        CALL io_push_value('n_imfreq',n_imfreq,numsp)
        CALL io_push_value('n_refreq',n_refreq,numsp)
        CALL io_push_value('ecut_imfreq [Ry]',ecut_imfreq,numsp)
        CALL io_push_value('ecut_refreq [Ry]',ecut_refreq,numsp)
        CALL io_push_value('wfreq_eta [Ry]',wfreq_eta,numsp)
        CALL io_push_value('n_secant_maxiter',n_secant_maxiter,numsp)
        CALL io_push_value('trev_secant [Ry]',trev_secant,numsp)
        CALL io_push_value('l_enable_lanczos',l_enable_lanczos,numsp)
        CALL io_push_value('l_enable_gwetot',l_enable_gwetot,numsp)
        CALL io_push_value('o_restart_time [min]',o_restart_time,numsp)
        CALL io_push_value('ecut_spectralf(1) [Ry]',ecut_spectralf(1),numsp)
        CALL io_push_value('ecut_spectralf(2) [Ry]',ecut_spectralf(2),numsp)
        CALL io_push_value('n_spectralf',n_spectralf,numsp)
        !
        CALL io_push_bar()
        !
     ENDIF
     !
     IF ( ANY(driver(:)==4) ) THEN
        !
        ! REPORT
        !
        CALL io_push_title('I/O Summary : westpp_control')
        !
        numsp=40
        CALL io_push_value('westpp_calculation',westpp_calculation,numsp)
        CALL io_push_value('westpp_range(1)',westpp_range(1),numsp)
        CALL io_push_value('westpp_range(2)',westpp_range(2),numsp)
        CALL io_push_value('westpp_format',westpp_format,numsp)
        CALL io_push_value('westpp_sign',westpp_sign,numsp)
        CALL io_push_value('westpp_n_pdep_eigen_to_use',westpp_n_pdep_eigen_to_use,numsp)
        CALL io_push_value('westpp_r0(1)',westpp_r0(1),numsp)
        CALL io_push_value('westpp_r0(2)',westpp_r0(2),numsp)
        CALL io_push_value('westpp_r0(3)',westpp_r0(3),numsp)
        CALL io_push_value('westpp_nr',westpp_nr,numsp)
        CALL io_push_value('westpp_rmax',westpp_rmax,numsp)
        CALL io_push_value('westpp_epsinfty',westpp_epsinfty,numsp)
        !
        CALL io_push_bar()
        !
     ENDIF
     !
     IF ( ANY(driver(:)==5) ) THEN
        !
        ! REPORT
        !
        CALL io_push_title('I/O Summary : server_control')
        !
        numsp=40
        CALL io_push_value('document',document,numsp)
        !
        CALL io_push_bar()
        !
     ENDIF
     !
     IF ( ANY(driver(:)==6) ) THEN
        !
        ! REPORT
        !
        CALL io_push_title("I/O Summary : wbse_init")
        !
        numsp = 23
        CALL io_push_value('wbse_init_calculation',wbse_init_calculation,numsp)
        !CALL io_push_value('which_bse_method', which_bse_method,numsp)
        !CALL io_push_value('n_pdep_eigen',n_pdep_eigen,numsp)
        CALL io_push_value('chi_kernel',chi_kernel,numsp)
        CALL io_push_value('which_spin_channel',which_spin_channel,numsp)
        !CALL io_push_value('l_use_localise_repr',l_use_localise_repr,numsp)
        CALL io_push_value('overlap_thr',overlap_thr,numsp)
        !CALL io_push_value('l_use_bisection_thr',l_use_bisection_thr,numsp)
        CALL io_push_value('localization',localization,numsp)
        CALL io_push_value('bisection_info',bisection_info,numsp)
        CALL io_push_value('wfc_from_qbox',wfc_from_qbox,numsp)
        !
        CALL io_push_bar()
        !
    ENDIF
    !
    IF ( ANY(driver(:)==7) ) THEN
        !
        ! REPORT
        !
        CALL io_push_title("I/O Summary : wbse_control")
        !
        !
        numsp=23
        !
        IF (l_davidson) THEN
            !
            CALL io_push_value('wbse_calculation',wbse_calculation,numsp)
            CALL io_push_value('solver',solver,numsp)
            CALL io_push_value('qp_correction',qp_correction,numsp)
            CALL io_push_value('n_liouville_eigen',n_liouville_eigen,numsp)
            CALL io_push_value('n_liouville_times',n_liouville_times,numsp)
            CALL io_push_value('n_liouville_maxiter',n_liouville_maxiter,numsp)
            CALL io_push_value('n_liouville_read_from_file',n_liouville_read_from_file,numsp)
            CALL io_push_value('l_bse_calculation',l_bse_calculation,numsp)
            !CALL io_push_value('l_diag_term_only',l_diag_term_only,numsp)
            CALL io_push_value('l_preconditioning',l_preconditioning,numsp)
            !CALL io_push_value('l_qp_correction',l_qp_correction,numsp)
            CALL io_push_value('spin_excitation', spin_excitation,numsp)
            CALL io_push_value('trev_liouville',trev_liouville,numsp)
            CALL io_push_value('trev_liouville_rel',trev_liouville_rel,numsp)
            CALL io_push_value('scissor_ope',scissor_ope,numsp)
            CALL io_push_value('epsinfty',epsinfty,numsp)
            !
        ENDIF
        !
        IF (l_lanzcos) THEN
            !
            CALL io_push_value('wbse_calculation',wbse_calculation,numsp)
            CALL io_push_value('solver',solver,numsp)
            CALL io_push_value('qp_correction',qp_correction,numsp)
            CALL io_push_value('ipol_input',ipol_input,numsp)
            CALL io_push_value('n_lanczos',n_lanczos,numsp)
            CALL io_push_value('l_bse_calculation',l_bse_calculation,numsp)
            CALL io_push_value('l_diag_term_only',l_diag_term_only,numsp)
            !CALL io_push_value('l_qp_correction',l_qp_correction,numsp)
            CALL io_push_value('spin_excitation', spin_excitation,numsp)
            CALL io_push_value('scissor_ope',scissor_ope,numsp)
            CALL io_push_value('epsinfty',epsinfty,numsp)
            CALL io_push_value('macropol_dfpt',macropol_dfpt,numsp)
            !
         ENDIF
         !
         CALL io_push_bar()
         !
    ENDIF
    !
!    IF ( ANY(driver(:)==8) ) THEN
!        !
!        ! REPORT
!        !
!        CALL io_push_title('I/O Summary : qbox_control')
!        !
!        numsp=23
!        IF ( nrowmax /=0 ) CALL io_push_value('nrowmax',nrowmax,numsp)
!        CALL io_push_value('xml_file',xml_file,numsp)
!        CALL io_push_value('xc',xc,numsp)
!        IF ( TRIM(xc) == 'PBE0' ) CALL io_push_value('alpha_pbe0',alpha_pbe0,numsp)
!        CALL io_push_value('wf_dyn',wf_dyn,numsp)
!        CALL io_push_value('blHF',blHF,numsp)
!        CALL io_push_value('btHF',btHF,numsp)
!        CALL io_push_value('amplitude',amplitude,numsp)
!        CALL io_push_value('nitscf',nitscf,numsp)
!        CALL io_push_value('nite',nite,numsp)
!        !
!        CALL io_push_bar()
!        !
!    ENDIF
    !
    IF ( ANY(driver(:)==8) ) THEN
        !
        ! REPORT
        !
        CALL io_push_title('I/O Summary : wbsepp_input')
        !
        numsp=30
        !
        IF (l_eig_decomp) THEN
           !
           CALL io_push_value('wbsepp_type',wbsepp_type,numsp)
           CALL io_push_value('n_liouville_read_from_file',n_liouville_read_from_file,numsp)
           CALL io_push_value('macropol_dfpt',macropol_dfpt,numsp)
           !
        ENDIF
        !
        IF (l_exc_rho_res_plot) THEN
           !
           CALL io_push_value('wbsepp_type',wbsepp_type,numsp)
           CALL io_push_value('n_liouville_read_from_file',n_liouville_read_from_file,numsp)
           CALL io_push_value('iexc_plot',iexc_plot,numsp)
           !
        ENDIF
        !
        IF (l_exc_plot ) THEN
           !
           CALL io_push_value('wbsepp_type',wbsepp_type,numsp)
           CALL io_push_value('n_liouville_read_from_file',n_liouville_read_from_file,numsp)
           CALL io_push_value('r0_input(1) [alat]',r0_input(1),numsp)
           CALL io_push_value('r0_input(2) [alat]',r0_input(2),numsp)
           CALL io_push_value('r0_input(3) [alat]',r0_input(3),numsp)
           CALL io_push_value('iexc_plot',iexc_plot,numsp)
           !
        ENDIF
        !
        IF (l_lz_spec) THEN
           !
           CALL io_push_value('wbsepp_type',wbsepp_type,numsp)
           CALL io_push_value('lziter_max', itermax,numsp)
           CALL io_push_value('lziter_cal', itermax0,numsp)
           CALL io_push_value('l_extrapolation', extrapolation,numsp)
           CALL io_push_value('w_start', start,numsp)
           CALL io_push_value('w_stop', end,numsp)
           CALL io_push_value('w_delta', increment,numsp)
           CALL io_push_value('w_unit', units,numsp)
           CALL io_push_value('ipol_index',ipol,numsp)
           CALL io_push_value('sym_opt',sym_op,numsp)
           CALL io_push_value('spin_channel',spin_channel,numsp)
           !
         ENDIF
        !
        CALL io_push_bar()
        !
     ENDIF
    !
  ENDIF
  !
  IF ( verbose .AND. mpime == root ) THEN
     !
     CALL json%initialize()
     CALL json%load_file(filename=TRIM(logfile))
     !
     CALL add_intput_parameters_to_json_file( num_drivers, driver, json )
     !
     OPEN( NEWUNIT=iunit, FILE=TRIM(logfile) )
     CALL json%print_file( iunit )
     CLOSE( iunit )
     CALL json%destroy()
     !
     !
  ENDIF
  !
  CALL stop_clock('fetch_input')
  !
END SUBROUTINE
