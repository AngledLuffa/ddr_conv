import os
import tempfile
import unittest

import simfile

def make_test_simfile(filename):
    text="""
#TITLE:Oil Ocean;
#SUBTITLE:HouseMasta Mix;
#ARTIST:Sonic 2;
#TITLETRANSLIT:;
#SUBTITLETRANSLIT:;
#ARTISTTRANSLIT:;
#GENRE:;
#CREDIT:;
#BANNER:OO-banner.png;
#BACKGROUND:OO-bg.png;
#LYRICSPATH:;
#CDTITLE:;
#MUSIC:OO.mp3;
#OFFSET:1.912;
#SAMPLESTART:15.340;
#SAMPLELENGTH:12.000;
#SELECTABLE:YES;
#BPMS:0.000=125.000;
#STOPS:;
#BGCHANGES:;
#KEYSOUNDS:;

//---------------dance-single - ----------------
#NOTES:
     dance-single:
     :
     Hard:
     7:
     0.450,0.472,0.232,0.111,0.162,278.000,23.000,11.000,0.000,0.000,0.000,0.450,0.472,0.232,0.111,0.162,278.000,23.000,11.000,0.000,0.000,0.000:
  // measure 1
0000
0100
0010
0000
,
0010
1000
0010
0100
0010
0100
0000
0000
;
//---------------dance-single - ----------------
#NOTES:
     dance-single:
     :
     Medium:
     5:
     0.363,0.446,0.212,0.131,0.000,218.000,21.000,13.000,0.000,0.000,0.000,0.363,0.446,0.212,0.131,0.000,218.000,21.000,13.000,0.000,0.000,0.000:


0000
0000
0010
0000
,
0000
0000
0010
0000
;
"""

    with open(filename, "w") as fout:
        fout.write(text)
    
class TestSimfile(unittest.TestCase):
    def setUp(self):
        self.test_file = tempfile.NamedTemporaryFile(suffix=".sm", delete=False)
        self.test_file.close()

    def tearDown(self):
        os.unlink(self.test_file.name)

    def test_read_sm(self):
        make_test_simfile(self.test_file.name)
        sim = simfile.read_sm_simfile(self.test_file.name)
        assert sim.offset == 1.912
        assert len(sim.bpms) == 1
        assert sim.bpms[0] == (0, 125)
        assert len(sim.stops) == 0
        assert len(sim.charts) == 2
        assert sim.num_beats() == 8

if __name__ == '__main__':
    unittest.main()

