import numpy as np
from numpy import ones


#per baseline, thephase
def unwrapxyphase(thephase,thestd,startchan):
    nchan=len(thephase)
    chans=np.arange(nchan)
    unwrapped=thephase+0
    segmentoffset=thephase+0
    segmentslope=0*thephase+0
    segment=np.zeros(len(thephase))#0 means unclassified
    order=np.argsort(abs(np.arange(nchan)-startchan))
    countsegments=0
    for i in order:#must classify in this order only#i is current channel number under investigation
        ineigh0=i-1
        ineigh1=(i+1)%nchan
        if segment[ineigh0] and segment[ineigh1]:#must merge to left and right; first merge to one that is closest, then unwrap merging of two islands
            r0=np.nonzero(segment==segment[ineigh0])[0]
            r1=np.nonzero(segment==segment[ineigh1])[0]
            if np.sum(1./thestd[r0])>np.sum(1./thestd[r1]):#LHS dominant unwrap left first
                #use offset and slope to predict
                extrap=i*segmentslope[ineigh0]+segmentoffset[ineigh0]
                unwrapped[i]=(thephase[i]-extrap+np.pi)%(2*np.pi)+extrap-np.pi
                segment[i]=segment[ineigh0]
                if True:
                    #now merge right with left
                    extrap1=i*segmentslope[ineigh1]+segmentoffset[ineigh1]
                    delta=(extrap1-extrap+np.pi)%(2*np.pi)+extrap-np.pi-extrap1
                    unwrapped[r1]+=delta
                    segment[r1]=segment[ineigh0]
                #recalculate segment slope and offset
                valid=np.nonzero(segment==segment[i])[0]
                #unwrapped[valid]=[chans[valid],ones(len(valid))]*[slope,offset]#1000x1=1000x2 2x1
                #[chans[valid],ones(len(valid))]T*unwrapped[valid]=[chans[valid],ones(len(valid))]T*[chans[valid],ones(len(valid))]*[slope,offset]# [2,1000][1000,1]=[2,1000][1000,2][2,1]
                w=1./thestd[valid]
                XT=np.array([chans[valid],ones(len(valid))])
                wX=XT.T*w[:,np.newaxis]
                #XT*(w*Y)=XT*(w*X)*C
                C=np.dot(np.linalg.pinv(np.dot(XT,wX)),np.dot(XT,(w*unwrapped[valid])[:,np.newaxis]))
                segmentslope[valid]=C[0]
                segmentoffset[valid]=C[1]

            else:#RHS dominant unwrap right first
                #use offset and slope to predict
                extrap=i*segmentslope[ineigh1]+segmentoffset[ineigh1]
                unwrapped[i]=(thephase[i]-extrap+np.pi)%(2*np.pi)+extrap-np.pi
                segment[i]=segment[ineigh1]
                if True:
                    #now merge right with left
                    extrap0=i*segmentslope[ineigh0]+segmentoffset[ineigh0]
                    delta=(extrap0-extrap+np.pi)%(2*np.pi)+extrap-np.pi-extrap0
                    unwrapped[r0]+=delta
                    segment[r0]=segment[ineigh1]
                #recalculate segment slope and offset
                valid=np.nonzero(segment==segment[i])[0]
                #unwrapped[valid]=[chans[valid],ones(len(valid))]*[slope,offset]#1000x1=1000x2 2x1
                #[chans[valid],ones(len(valid))]T*unwrapped[valid]=[chans[valid],ones(len(valid))]T*[chans[valid],ones(len(valid))]*[slope,offset]# [2,1000][1000,1]=[2,1000][1000,2][2,1]
                w=1./thestd[valid]
                XT=np.array([chans[valid],ones(len(valid))])
                wX=XT.T*w[:,np.newaxis]
                #XT*(w*Y)=XT*(w*X)*C
                C=np.dot(np.linalg.pinv(np.dot(XT,wX)),np.dot(XT,(w*unwrapped[valid])[:,np.newaxis]))
                segmentslope[valid]=C[0]
                segmentoffset[valid]=C[1]
        elif segment[ineigh0]:#must merge this sample with left only using slopes and offsets of islands;update slope and offset
            #use offset and slope to predict
            extrap=i*segmentslope[ineigh0]+segmentoffset[ineigh0]
            unwrapped[i]=(thephase[i]-extrap+np.pi)%(2*np.pi)+extrap-np.pi
            segment[i]=segment[ineigh0]
            #recalculate segment slope and offset
            valid=np.nonzero(segment==segment[i])[0]
            #unwrapped[valid]=[chans[valid],ones(len(valid))]*[slope,offset]#1000x1=1000x2 2x1
            #[chans[valid],ones(len(valid))]T*unwrapped[valid]=[chans[valid],ones(len(valid))]T*[chans[valid],ones(len(valid))]*[slope,offset]# [2,1000][1000,1]=[2,1000][1000,2][2,1]
            w=1./thestd[valid]
            XT=np.array([chans[valid],ones(len(valid))])
            wX=XT.T*w[:,np.newaxis]
            #XT*(w*Y)=XT*(w*X)*C
            C=np.dot(np.linalg.pinv(np.dot(XT,wX)),np.dot(XT,(w*unwrapped[valid])[:,np.newaxis]))
            segmentslope[valid]=C[0]
            segmentoffset[valid]=C[1]
        elif segment[ineigh1]:#must merge this sample with right only using slopes and offsets of islands;update slope and offset
            #use offset and slope to predict
            extrap=i*segmentslope[ineigh1]+segmentoffset[ineigh1]
            unwrapped[i]=(thephase[i]-extrap+np.pi)%(2*np.pi)+extrap-np.pi
            segment[i]=segment[ineigh1]
            #recalculate segment slope and offset
            valid=np.nonzero(segment==segment[i])[0]
            #unwrapped[valid]=[chans[valid],ones(len(valid))]*[slope,offset]#1000x1=1000x2 2x1
            #[chans[valid],ones(len(valid))]T*unwrapped[valid]=[chans[valid],ones(len(valid))]T*[chans[valid],ones(len(valid))]*[slope,offset]# [2,1000][1000,1]=[2,1000][1000,2][2,1]
            w=1./thestd[valid]
            XT=np.array([chans[valid],ones(len(valid))])
            wX=XT.T*w[:,np.newaxis]
            #XT*(w*Y)=XT*(w*X)*C
            C=np.dot(np.linalg.pinv(np.dot(XT,wX)),np.dot(XT,(w*unwrapped[valid])[:,np.newaxis]))
            segmentslope[valid]=C[0]
            segmentoffset[valid]=C[1]
        else:#both sides still unclassified; make 'new island' with zero slope, and unwrapped=wrapped here
            countsegments+=1
            segment[i]=countsegments
            unwrapped[i]=thephase[i]

    return unwrapped,segmentslope[0],segmentoffset[0]


def mattieu(x, gain, phase_std):
    M, N = x.shape
    if gain is None:
        gain = np.ones(N)
    gain_without_zeros = np.where(gain <= 0, 100, gain)
    thestd = gain_without_zeros * phase_std
    slope = np.empty(M)
    for m in range(M):
        _, slope[m], _ = unwrapxyphase(np.angle(x[m]), thestd, startchan=600)
    return slope
