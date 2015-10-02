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




    instr 1
          a1  oscili  gkamp1, 1 * 40, 1           
          a2  oscili  gkamp2, 2 * 40, 1           
          a3  oscili  gkamp3, 3 * 40, 1           
          a4  oscili  gkamp4, 4 * 40, 1           
          a5  oscili  gkamp5, 5 * 40, 1           
          a6  oscili  gkamp6, 6 * 40, 1           
          a7  oscili  gkamp7, 7 * 40, 1           
          a8  oscili  gkamp8, 8 * 40, 1           
          a9  oscili  gkamp9, 9 * 40, 1               
          a10  oscili gkamp10,10* 40, 1           
          a11  oscili gkamp11,11* 40, 1           
          a12  oscili gkamp12,12* 40, 1           
          a13  oscili gkamp13,13* 40, 1           
          a14  oscili gkamp14,14* 40, 1           
          a15  oscili gkamp15,15* 40, 1           
          a16  oscili gkamp16,16* 40, 1           
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
