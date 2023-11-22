import katdal
import katpoint
import numpy as np
from katsdpcal.plotting import *
from katdal.spectral_window import SpectralWindow
from katpoint import (rad2deg, deg2rad, lightspeed, wrap_angle, RefractionCorrection)
from scikits.fitting import ScatterFit, GaussianFit
import dask.array as da
import ephem
import pytest
from katsdpcalproc import pointing


middle_time=1691795333.43713
temperature=14.7
humidity=29.1
pressure=897
track_duration=24
NUM_CHUNKS=5
pols=["h"]

centre_freq=1284000000.0
bandwidth=856000000.0
no_channels=1000
channel_freqs=centre_freq +(np.arange(no_channels) - no_channels/2)*(bandwidth/no_channels)
chunk_freqs = channel_freqs.reshape(NUM_CHUNKS, -1).mean(axis=1)
target=katpoint.Target(body="J1939-6342, radec bfcal single_accumulation, 19:39:25.03, -63:42:45.6")

# Maximum distance of offset from target, in degrees
max_extent = 1.0
num_pointings = 8
# Build up sequence of pointing offsets running linearly in x and y directions
scan = np.linspace(-max_extent, max_extent, num_pointings // 2)
offsets_along_x = np.c_[scan, np.zeros_like(scan)]
offsets_along_y = np.c_[np.zeros_like(scan), scan]
offsets = np.r_[offsets_along_y, offsets_along_x]

ants=[katpoint.Antenna('m000','-30:42:39.8','21:26:38.0',1086.6,diameter=15,beamwidth=1.22,pointing_model="0:04:20.6,0,0:01:14.2,0:02:58.5,0:00:05.1,0:00:00.4,0:20:04.1,-0:00:34.5,0,0,-0:03:10.0,0,0,0,0,0,0,0,0,0,0,0",delay_model="-8.264,-207,8.6,212.6,212.6,1"),katpoint.Antenna('m001','-30:42:39.8','21:26:38.0',1086.6,diameter=15,beamwidth=1.22,pointing_model="0:04:15.6,0,0:01:09.2,0:01:58.5,0:00:05.1,0:00:00.4,0:16:04.1,-0:00:34.5,0,0,-0:03:10.0,0,0,0,0,0,0,0,0,0,0,0",delay_model="-8.264,-207,8.6,212.6,212.6,1")]
existing_az_el_adjust=np.zeros((len(ants),2))

## Creating gains, numpy array shape (no.offsets, no.polarisations, no.antennas)
weights=np.ones(10)
just_gains=[]
for i in range(0,len(offsets)):
    gg=[]
    for j in pols:
        gg.append(np.array(np.ones(len(ants))))
    just_gains.append(np.array(gg))  
just_gains=np.array(just_gains)




## Creating bp_gains, numpy array of shape (no.offsets, no.freqs, no.polarisations, no.antennas)

def g_o_g(offsets,ants,channel_freqs):
    bp_gains3=[]
    for i in range(0,len(offsets)):
        bp_gains2=[]
        for f in range(0, len(channel_freqs)):
            bp_gains1=[]
            for pol in pols:
                ex_width=[]
                for a, ant in enumerate(ants):
                    # expected widths for each frequency channel
                    expected_width = rad2deg(ant.beamwidth * lightspeed /
                                                         channel_freqs[f]/ ant.diameter)
                    # Convert power beamwidth to gain / voltage beamwidth
                    expected_width = np.sqrt(2.0) * expected_width
                    # XXX This assumes we are still using default ant.beamwidth of 1.22
                    # and also handles larger effective dish diameter in H direction
                    expected_width = (0.8 * expected_width, 0.9 * expected_width)
                    ex_width.append(expected_width)
                    
                gains=[]
                for k in ex_width:
                    new_beam=pointing.BeamPatternFit((0,0),k,1.0)
                    g=new_beam(x=offsets[i].T)
                    gains.append(np.array(g))

                gains=np.array(gains).T
                bp_gains1.append(np.array(gains))
            bp_gains2.append(np.array(bp_gains1))
        bp_gains3.append(np.array(bp_gains2))
    return np.array(bp_gains3)


# In[18]:




bp_gains=g_o_g(offsets,ants,channel_freqs)
data_points= pointing.get_offset_gains(bp_gains,just_gains,offsets,NUM_CHUNKS,ants,track_duration,centre_freq,bandwidth,no_channels)
beams=pointing.beam_fit(data_points,NUM_CHUNKS,ants)
   
pointing_offsets=pointing.calc_pointing_offsets(ants,middle_time,temperature,humidity,pressure,beams,target,existing_az_el_adjust)





# In[23]:




## Test that length of data_points equals the legnth of antenna list
def test_get_offset_gains_len():
    assert len(data_points)== len(ants)


# In[24]:



## Test that incorrect shape of gains will raise Index Error
def test_get_offset_gains_shape():
     with pytest.raises(IndexError):
        pointing.get_offset_gains(bp_gains[0],just_gains[0],offsets,NUM_CHUNKS,ants,track_duration,centre_freq,bandwidth,no_channels)  
## Raise error if NUM_CHUNKS is not multiple of no_channels
def test_get_offset_gains_multiple():
     with pytest.raises(NotMUltipleError):
        pointing.get_offset_gains(bp_gains[0],just_gains,offsets,3,ants,track_duration,centre_freq,bandwidth,no_channels)  
## Test legnth of each data_points element
def test_get_offset_gains_len2():
    for i in range (0, len(data_points)):
        assert len(list(data_points.items())[i][1])== NUM_CHUNKS*len(offsets) 
        for j in range(0,NUM_CHUNKS*len(offsets)):
            assert len(list(data_points.items())[i][1][j])== 5
## Test that inputting a float in place of a list for gains raises Type Error
def test_get_offset_gains_type():
    with pytest.raises(TypeError):
        pointing.get_offset_gains(9,just_gains,offsets,NUM_CHUNKS,ants,track_duration,centre_freq,bandwidth,no_channels)
## Testing that the output of beam_fit are of type BeamPatternFit
def test_beam_fit_type():
    assert len(beams)==len(ants)
    for i in ants:
        assert type(beams[i.name][0]) and type(beams[i.name][-1]) == type(None)
        for j in range(1,NUM_CHUNKS-1):
            assert type(beams[i.name][j]) == pointing.BeamPatternFit
## Multiple small type errors for calc_pointing_offsets
def test_calc_pointing_offsets_random():
    with pytest.raises(NotUnixTime):
        po=pointing.calc_pointing_offsets(ants,1220,temperature,humidity,pressure,beams,target,existing_az_el_adjust)
    with pytest.raises(TypeError):
        po=pointing.calc_pointing_offsets(ants,'12:20',temperature,humidity,pressure,beams,target,existing_az_el_adjust) 
    with pytest.raises(NotKatpointTarget):
        po=pointing.calc_pointing_offsets(ants,middle_time,temperature,humidity,pressure,beams,'target',existing_az_el_adjust) 
## Test that the legnth of each pointing offset solution =10 (5 sets of (x,y) coordinates)
def test_calc_pointing_offsets_len():
    assert len(pointing_offsets)==len(ants)
    for i in range (0, len(pointing_offsets)):
        assert len(list(pointing_offsets.items())[i][1])== 10



# In[25]:



## Compare widths of simulated primary beams from beam_fit and original beam object
def test_fit_primary_beams():

    
    
    def get_widths(offsets,ants,chunk_freqs):
        for i in range(0,len(offsets)):
            compare_ex_width={}
            for a, ant in enumerate(ants):
                ex_width=[]
                
                for chunk in range(0, NUM_CHUNKS):
                    

                    expected_width = rad2deg(ant.beamwidth * lightspeed /
                                                         chunk_freqs[chunk]/ ant.diameter)
                    # Convert power beamwidth to gain / voltage beamwidth
                    expected_width = np.sqrt(2.0) * expected_width
                    # XXX This assumes we are still using default ant.beamwidth of 1.22
                    # and also handles larger effective dish diameter in H direction
                    expected_width = (0.8 * expected_width, 0.9 * expected_width)
                    ex_width.append(expected_width)
                    compare_ex_width[str(ant.name)]=ex_width

        return compare_ex_width
    
    comp_widths=get_widths(offsets,ants,chunk_freqs)
    
    #### Feeding simulated data_points into beam_fit function
    
    j=pointing.beam_fit(data_points,NUM_CHUNKS,ants)
    
    ###Comparing output of beam_fit to the original beam object
    for z in range(1,NUM_CHUNKS-1):
        for x in j.keys():
            assert j[x][z].expected_width==pytest.approx(comp_widths[x][z],abs=0.0001)

