from collections import OrderedDict
import os

def numbered_list(value, key_name):
    """
    Converts a numbered list of the form X1=Y1, X2=Y2, ...
    into [(X1, Y1), (X2, Y2), ...]
    Useful for reading BPMS and STOPS
    """
    pairs = []
    if value is None:
        return pairs
    value = value.strip()
    if not value:
        return pairs
    items = value.split(",")
    for item in items:
        pieces = item.split("=")
        if len(pieces) != 2:
            raise ValueError("Illegal format for %s" % key_name)
        pieces = [float(x.strip()) for x in pieces]
        pairs.append(tuple(pieces))
    return pairs

class StepChart(object):
    def __init__(self, chart):
        chart = chart.strip()
        traits = chart.split(":")
        self.game = traits[0]
        # a possibly user-supplied description of the steps,
        # often just nothing
        self.description = traits[1]
        # beginner, hard, etc
        self.difficulty = traits[2]
        # numeric value of the difficulty
        self.rating = traits[3]
        self.radar = traits[4]
        measures = [x.strip() for x in traits[5].strip().split(",")]
        # TODO: could further pre-process the measures into steps
        self.measures = measures

class Simfile(object):
    def __init__(self, pairs, charts):
        """
        Keeps track of a list of key/value pairs for the simfile.
        Also keeps offset/stops/bpms separately to make processing easier.
        """
        # TODO: come up with a convenient way to maintain the order of
        # the items not being kept in the dict.  Perhaps leave empty
        # values in the dict, for example
        self.pairs = OrderedDict(pairs)
        self.charts = [StepChart(x) for x in charts]

        if 'offset' not in pairs:
            raise ValueError('Simfile has no offset')
        else:
            self.offset = float(pairs.get('offset'))
            del self.pairs['offset']
            
        if 'bpms' not in pairs:
            raise ValueError('Simfile has no bpm')
        else:
            self.bpms = numbered_list(pairs.get('bpms'), 'bpms')
            del self.pairs['bpms']

        if 'stops' not in pairs:
            self.stops = []
        else:
            self.stops = numbered_list(pairs.get('stops', ""), 'stops')
            del self.pairs['stops']

        if 'music' not in pairs:
            raise ValueError('Simfile has no music!')

    def update_bpms(self, offset, bpms, stops):
        """
        Given new text for the bpms and stops, updates the internal timing.
        Updates both the pairs representation and the bpms/stops timing.
        """
        self.offset = float(offset)
        self.bpms = numbered_list(bpms, "bpms")
        self.stops = numbered_list(stops, "stops")

    def beat(self, time):
        """
        Converts a time to a beat
        """
        if time == -self.offset:
            return 0.0
        if time < -self.offset:
            return (time + self.offset) / (60. / self.bpms[0][1])
        # has to be a positive beat at this point.
        # need to account for bpm changes and stops
        # TODO: account for changes & stops
        for previous, current in zip(self.bpms[:-1], self.bpms[1:]):
            raise RuntimeError("TODO")
        if len(self.stops) > 0:
            raise RuntimeError("TODO")
        return (time + self.offset) / (60. / self.bpms[0][1])

    def time(self, beat):
        """
        Converts a beat to a time.
        """
        if beat == 0:
            return -self.offset
        if beat < 0:
            return -self.offset + beat * 60. / self.bpms[0][1]
        if beat > 0:
            time = -self.offset
            current_beat = 0
            # TODO: test bpm changes
            for previous, current in zip(self.bpms[:-1], self.bpms[1:]):
                if current[0] > beat:
                    time = time + (beat - current_beat) * (60.0 / previous[1])
                    current_beat = beat
                    break
                else:
                    time = time + (current[0] - current_beat) * (60.0 / previous[1])
                    current_beat = current[0]
            # any leftover time comes from the last bpm
            # (which may be the only bpm)
            if current_beat < beat:
                bpm = self.bpms[-1]
                time = time + (beat - current_beat) * (60.0 / bpm[1])
            # TODO: test stops
            for stop in self.stops:
                if stop[0] < beat:
                    time = time + stop[1]
            return time

def read_sm_simfile(filename):
    # TODO: be more forgiving?
    #   - For example, can allow multiple , in a row
    #   - can also have some tags which are expected to be exactly one
    #     line for which ; is optional, as that mistake also
    #     occasionally happens
    pairs = OrderedDict()
    with open(filename, encoding='utf-8') as fin:
        lines = fin.readlines()
    key = None
    value = None
    charts = []
    for line in lines:
        comment = line.find("//")
        if comment >= 0:
            line = line[:comment]
        line = line.strip()
        if key is None:
            key_start = line.find("#")
            if key_start < 0:
                # still not in a key, so no need to save the text
                continue
            key_start = key_start + 1
            key_end = line[key_start:].find(":") + key_start
            if key_end < 0:
                raise RuntimeError("Key spanned multiple lines")
            if key_start == key_end:
                raise RuntimeError("Empty key #:")
            key = line[key_start:key_end]
            line = line[key_end+1:]
        # now we are in a key.  save text and keep going
        if value is None:
            value = ""
        value_end = line.find(";")
        if value_end >= 0:
            value = value + line[:value_end]
            if key.lower() == 'notes':
                charts.append(value)
            else:
                pairs[key.lower()] = value

            key = None
            value = None
        else:
            value = value + "\n" + line

    if key is not None:
        raise ValueError("Unfinished key '%s' in simfile %s " % (key, filename))

    # expand music path
    if 'music' not in pairs:
        raise ValueError("Could not find music for simfile %s" % filename)
    else:
        music_path = os.path.join(os.path.split(filename)[0], filename)
        pairs['music'] = music_path

    return Simfile(pairs, charts)


def read_simfile(filename):
    extension = os.path.splitext(filename)[1]
    try:
        if extension == '.sm':
            return read_sm_simfile(filename)
    except ValueError as e:
        raise ValueError("Could not read %s" % filename) from e
    raise ValueError("Cannot read simfiles with extension %s" % extension)
