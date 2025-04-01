
 &XC
   &XC_FUNCTIONAL
     &LYP
       SCALE_C 0.81 ! mixing parameters explained above
     &END
     &BECKE88
       SCALE_X 0.72
     &END
     &XALPHA
       SCALE_X 0.08
     &END
   &END XC_FUNCTIONAL
   &HF
     &SCREENING
       EPS_SCHWARZ 1.0E-10
     &END
     &MEMORY
       MAX_MEMORY 5
     &END
     FRACTION 0.20
   &END
 &END XC
