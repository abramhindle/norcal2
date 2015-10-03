sr = 44100
kr = 441
ksmps = 100
nchnls = 2
0dbfs = 1.0

gihandle OSCinit 7770


gkamp1  init 0
gkamp2  init 0
gkamp3  init 0
gkamp4  init 0
gkamp5  init 0
gkamp6  init 0
gkamp7  init 0
gkamp8  init 0
gkamp9  init 0
gkamp10 init 0
gkamp11 init 0
gkamp12 init 0
gkamp13 init 0
gkamp14 init 0
gkamp15 init 0
gkamp16 init 0


gkport init 1/90

    instr 1
        ifn = p4
        kf1 init 0
        kf2 init 0
        kf3 init 0
        kf4 init 0
        kf5 init 0
        kf6 init 0
        kf7 init 0
        kf8 init 0
        kf9 init 0
        kf10 init 0
        kf11 init 0
        kf12 init 0
        kf13 init 0
        kf14 init 0
        kf15 init 0
        kf16 init 0
        kf1  portk gkamp1, gkport
        kf2  portk gkamp2, gkport
        kf3  portk gkamp3, gkport
        kf4  portk gkamp4, gkport
        kf5  portk gkamp5, gkport
        kf6  portk gkamp6, gkport
        kf7  portk gkamp7, gkport
        kf8  portk gkamp8, gkport
        kf9  portk gkamp9, gkport
        kf10 portk gkamp10, gkport
        kf11 portk gkamp11, gkport
        kf12 portk gkamp12, gkport
        kf13 portk gkamp13, gkport
        kf14 portk gkamp14, gkport
        kf15 portk gkamp15, gkport
        kf16 portk gkamp16, gkport
        
          a1  oscili  kf1, 1 * 40, ifn           
          a2  oscili  kf2, 2 * 40, ifn           
          a3  oscili  kf3, 3 * 40, ifn           
          a4  oscili  kf4, 4 * 40, ifn           
          a5  oscili  kf5, 5 * 40, ifn           
          a6  oscili  kf6, 6 * 40, ifn           
          a7  oscili  kf7, 7 * 40, ifn           
          a8  oscili  kf8, 8 * 40, ifn           
          a9  oscili  kf9, 9 * 40, ifn               
          a10  oscili kf10,10* 40, ifn           
          a11  oscili kf11,11* 40, ifn           
          a12  oscili kf12,12* 40, ifn           
          a13  oscili kf13,13* 40, ifn           
          a14  oscili kf14,14* 40, ifn           
          a15  oscili kf15,15* 40, ifn           
          a16  oscili kf16,16* 40, ifn           
          aout = a1 +  a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15 + a16
          outs aout, aout
    endin
    

    instr oscmix       
        kf1 init 0
        kf2 init 0
        kf3 init 0
        kf4 init 0
        kf5 init 0
        kf6 init 0
        kf7 init 0
        kf8 init 0
        kf9 init 0
        kf10 init 0
        kf11 init 0
        kf12 init 0
        kf13 init 0
        kf14 init 0
        kf15 init 0
        kf16 init 0
      nxtmsg:           
        kk  OSClisten gihandle, "/fft/sbins", "ffffffffffffffff", kf1, kf2, kf3, kf4, kf5, kf6, kf7, kf8, kf9, kf10, kf11, kf12, kf13, kf14, kf15, kf16
      if (kk == 0) goto ex
        gkamp1  =  1000 * kf1 
        gkamp2  =  1000 * kf2  
        gkamp3  =  1000 * kf3  
        gkamp4  =  1000 * kf4
        gkamp5  =  1000 * kf5
        gkamp6  =  1000 * kf6  
        gkamp7  =  1000 * kf7  
        gkamp8  =  1000 * kf8
        gkamp9  =  1000 * kf9
        gkamp10 =  1000 * kf10
        gkamp11 =  1000 * kf11  
        gkamp12 =  1000 * kf12  
        gkamp13 =  1000 * kf13  
        gkamp14 =  1000 * kf14
        gkamp15 =  1000 * kf15
        gkamp16 =  1000 * kf16  
        kgoto nxtmsg
      ex:
    endin
