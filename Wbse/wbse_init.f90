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
PROGRAM wbse_init
  !-----------------------------------------------------------------------
  !
  ! This is the main program that calculates the static screening.
  !
  use nvtx
  USE check_stop,           ONLY : check_stop_init
  USE mp_global,            ONLY : mp_startup,mp_global_end
  USE west_environment,     ONLY : west_environment_start,west_environment_end
  !
  IMPLICIT NONE
  !
  CHARACTER(LEN=9) :: code = 'WBSE_INIT'
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
  CALL west_environment_start( code )
  !
  call nvtxStartRange("WBSE_INIT")
  !!!!!!!!!!!!!!!!!!!!!!
  CALL west_readin( code )
  !
  call nvtxStartRange("INIT_SETUP")
    CALL wbse_init_setup( )
  call nvtxEndRange
  !
  call nvtxStartRange("CALC_TAU")
    CALL calc_tau( )
  call nvtxEndRange
  !
  call nvtxStartRange("EXX_UNGO")
    CALL exx_ungo( )
  call nvtxEndRange
  !!!!!!!!!!!!!!!!!!!!!!
  call nvtxEndRange
  !
  CALL clean_scratchfiles( )
  !
  CALL west_print_clocks( )
  !
  CALL west_environment_end( code )
  !
  CALL mp_global_end( )
  !
END PROGRAM
