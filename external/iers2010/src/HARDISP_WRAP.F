      SUBROUTINE HARDISP_WRAP (YEAR,MONTH,DAY,HOUR,MINUTE,SECOND,
     .  N,SAMPLE,BLQAMP,BLQPHASE,DU,DS,DW)
*+
*  HARDISP_WRAP is a project-derived callable version of the IERS
*  Conventions HARDISP.F program.  The source file and modified routine are
*  renamed as required by the IERS license.  The command-line,
*  standard-input, and standard-output boundary is replaced by array
*  arguments; the tidal expansion still uses the unchanged official ADMINT
*  and RECURS computational routines.
*
*  BLQAMP and BLQPHASE use the upstream three-component input order:
*  vertical, west, and south.  The output arrays are the corresponding dU,
*  dS, and dW series in metres.  This derived routine is not distributed by
*  or endorsed by the IERS Conventions Center.
*
*  Source basis:
*  IERS Conventions (2010), Chapter 7, HARDISP.F, ADMINT.F, and RECURS.F.
*-----------------------------------------------------------------------

      IMPLICIT NONE

      INTEGER YEAR,MONTH,DAY,HOUR,MINUTE,SECOND,N
      INTEGER I,IRHI,IRLI,NP,NTOUT,NT,NTIN,NL
      INTEGER IT,IDT,MDAY
      PARAMETER (NT=342)
      PARAMETER (NTIN=11)
      PARAMETER (NL=600)

      REAL AMP,AS,AW,AZ,DSR,DWR,DZR,HCS,HCW,HCZ,PHASE,WF
      DOUBLE PRECISION BLQAMP,BLQPHASE,DU,DS,DW
      DOUBLE PRECISION F,PZ,PS,PW,SCR
      DOUBLE PRECISION SAMPLE
      DOUBLE PRECISION DR,PI

      DIMENSION BLQAMP(3,NTIN),BLQPHASE(3,NTIN)
      DIMENSION DU(*),DS(*),DW(*)
      DIMENSION IT(5),IDT(6,NTIN),AMP(NTIN),PHASE(NTIN)
      DIMENSION AZ(NT),PZ(NT),HCZ(2*NT)
      DIMENSION AS(NT),PS(NT),HCS(2*NT)
      DIMENSION AW(NT),PW(NT),HCW(2*NT)
      DIMENSION DZR(NL),DSR(NL),DWR(NL)
      DIMENSION F(NT),SCR(3*NT),WF(NT)
      COMMON/DATE/IT
      PARAMETER (PI=3.1415926535897932384626433D0)
      DATA DR/0.01745329252D0/

*  Cartwright-Tayler numbers of tides used in Scherneck lists:
*      M2, S2, N2, K2, K1, O1, P1, Q1, Mf, Mm, Ssa
      DATA IDT/
     .  2, 0, 0, 0, 0, 0,   2, 2,-2, 0, 0, 0,   2,-1, 0, 1, 0, 0,
     .  2, 2, 0, 0, 0, 0,   1, 1, 0, 0, 0, 0,   1,-1, 0, 0, 0, 0,
     .  1, 1,-2, 0, 0, 0,   1,-2, 0, 1, 0, 0,   0, 2, 0, 0, 0, 0,
     .  0, 1, 0,-1, 0, 0,   0, 0, 2, 0, 0, 0/

      IF(N.LE.0) RETURN

      IT(1)=YEAR
      IT(2)=DAY+MDAY(YEAR,MONTH)
      IT(3)=HOUR
      IT(4)=MINUTE
      IT(5)=SECOND

*  ADMINT expects the standard Scherneck phases as negative lags.
      DO I=1,NTIN
         AMP(I)=REAL(BLQAMP(1,I))
         PHASE(I)=-REAL(BLQPHASE(1,I))
      ENDDO
      CALL ADMINT(AMP,IDT,PHASE,AZ,F,PZ,NTIN,NTOUT)

      DO I=1,NTIN
         AMP(I)=REAL(BLQAMP(2,I))
         PHASE(I)=-REAL(BLQPHASE(2,I))
      ENDDO
      CALL ADMINT(AMP,IDT,PHASE,AW,F,PW,NTIN,NTOUT)

      DO I=1,NTIN
         AMP(I)=REAL(BLQAMP(3,I))
         PHASE(I)=-REAL(BLQPHASE(3,I))
      ENDDO
      CALL ADMINT(AMP,IDT,PHASE,AS,F,PS,NTIN,NTOUT)

*  Normalize frequencies and convert phases to radians, as in HARDISP.F.
      DO I=1,NTOUT
         PZ(I)=DR*PZ(I)
         PS(I)=DR*PS(I)
         PW(I)=DR*PW(I)
         F(I)=SAMPLE*PI*F(I)/43200.D0
         WF(I)=F(I)
      ENDDO

*  RECURS works in blocks of NL samples and preserves the official recursive
*  evaluation for both short single-epoch and longer evenly spaced series.
      IRLI=1
 10   IRHI=MIN(IRLI+NL-1,N)
      NP=IRHI-IRLI+1

      DO I=1,NT
         HCZ(2*I-1)=AZ(I)*DCOS(PZ(I))
         HCZ(2*I)=-AZ(I)*DSIN(PZ(I))
         HCS(2*I-1)=AS(I)*DCOS(PS(I))
         HCS(2*I)=-AS(I)*DSIN(PS(I))
         HCW(2*I-1)=AW(I)*DCOS(PW(I))
         HCW(2*I)=-AW(I)*DSIN(PW(I))
      ENDDO
      CALL RECURS(DZR,NP,HCZ,NTOUT,WF,SCR)
      CALL RECURS(DSR,NP,HCS,NTOUT,WF,SCR)
      CALL RECURS(DWR,NP,HCW,NTOUT,WF,SCR)

      DO I=1,NP
         DU(IRLI+I-1)=DBLE(DZR(I))
         DS(IRLI+I-1)=DBLE(DSR(I))
         DW(IRLI+I-1)=DBLE(DWR(I))
      ENDDO

      IF(IRHI.EQ.N) RETURN
      IRLI=IRHI+1
      DO I=1,NT
         PZ(I)=DMOD(PZ(I)+NP*F(I),2.D0*PI)
         PS(I)=DMOD(PS(I)+NP*F(I),2.D0*PI)
         PW(I)=DMOD(PW(I)+NP*F(I),2.D0*PI)
      ENDDO
      GO TO 10

* Finished.

*+----------------------------------------------------------------------
*
*  Copyright (C) 2008
*  IERS Conventions Center
*
*  ==================================
*  IERS Conventions Software License
*  ==================================
*
*  NOTICE TO USER:
*
*  BY USING THIS SOFTWARE YOU ACCEPT THE FOLLOWING TERMS AND CONDITIONS
*  WHICH APPLY TO ITS USE.
*
*  1. The Software is provided by the IERS Conventions Center ("the
*     Center").
*
*  2. Permission is granted to anyone to use the Software for any
*     purpose, including commercial applications, free of charge,
*     subject to the conditions and restrictions listed below.
*
*  3. You (the user) may adapt the Software and its algorithms for your
*     own purposes and you may distribute the resulting "derived work"
*     to others, provided that the derived work complies with the
*     following requirements:
*
*     a) Your work shall be clearly identified so that it cannot be
*        mistaken for IERS Conventions software and that it has been
*        neither distributed by nor endorsed by the Center.
*
*     b) Your work (including source code) must contain descriptions of
*        how the derived work is based upon and/or differs from the
*        original Software.
*
*     c) The name(s) of all modified routine(s) that you distribute
*        shall be changed.
* 
*     d) The origin of the IERS Conventions components of your derived
*        work must not be misrepresented; you must not claim that you
*        wrote the original Software.
*
*     e) The source code must be included for all routine(s) that you
*        distribute.  This notice must be reproduced intact in any
*        source distribution. 
*
*  4. In any published work produced by the user and which includes
*     results achieved by using the Software, you shall acknowledge
*     that the Software was used in obtaining those results.
*
*  5. The Software is provided to the user "as is" and the Center makes
*     no warranty as to its use or performance.   The Center does not
*     and cannot warrant the performance or results which the user may
*     obtain by using the Software.  The Center makes no warranties,
*     express or implied, as to non-infringement of third party rights,
*     merchantability, or fitness for any particular purpose.  In no
*     event will the Center be liable to the user for any consequential,
*     incidental, or special damages, including any lost profits or lost
*     savings, even if a Center representative has been advised of such
*     damages, or for any claim by any third party.
*
*  Correspondence concerning IERS Conventions software should be
*  addressed as follows:
*
*                     Gerard Petit
*     Internet email: gpetit[at]bipm.org
*     Postal address: IERS Conventions Center
*                     Time, frequency and gravimetry section, BIPM
*                     Pavillon de Breteuil
*                     92312 Sevres  FRANCE
*
*     or
*
*                     Brian Luzum
*     Internet email: brian.luzum[at]usno.navy.mil
*     Postal address: IERS Conventions Center
*                     Earth Orientation Department
*                     3450 Massachusetts Ave, NW
*                     Washington, DC 20392
*
*
*-----------------------------------------------------------------------
      END
