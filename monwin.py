from __future__ import print_function, division, absolute_import

import numpy as np
import matplotlib.pylab as plt

#import parse_cm as pcm
import copy
import collections

from chandra_aca import star_probs, transform
import agasc
from Quaternion import Quat, normalize
import Ska.quatutil as quatutil
import astropy.units as u
from astropy.table import Table
from datetime import datetime, timedelta


__all__ = ['get_dwells', 'monitor_window_stats', 'scheduler',
           'dwells_test', 'stars_test', 'monwin_test']


# file containing coordinates of predefined locations,
# expected to be found in the working directory
PREDEFINED = "predefined_locations.txt"


def add_delta(date, *args):
    """
    Example
    =======

    >>> date = '2017:007:23:45:36.709'
    >>> delta = '00:05:00.000'
    #>>> add_delta(date, delta)
    '2017:007:23:50:36.709'

    >>> delta = '001:00:00:00.000'
    >>> add_delta(date, delta)
    '2017:008:23:45:36.709'

    :date: datetime.datetime
    :*args: str with format
    :rtype: datetime.datetime
    """
    for arg in args:
        date = datetime.strptime(date, "%Y:%j:%H:%M:%S.%f")
        delta = format_to_timedelta(arg)
        if '-' in arg:
            date = (date - delta).strftime("%Y:%j:%H:%M:%S.%f")
        else:
            date = (date + delta).strftime("%Y:%j:%H:%M:%S.%f")
    return date[:-3]


def format_to_timedelta(date_time):
    """
    Example
    =======

    >>> date_time = '000:00:01:36.434'
    >>> format_to_timedelta(date_time)
    datetime.timedelta(0, 96, 434000) # days, seconds, microseconds

    >>> date_time = '00:05:00.000'
    >>> format_to_timedelta(date_time)
    datetime.timedelta(0, 300)

    >>> date_time = '0.0010'
    >>> format_to_timedelta(date_time)
    datetime.timedelta(0, 0, 1000)

    ::rtype:: datetime.timedelta
    """
    labels = ['microseconds', 'seconds', 'minutes', 'hours', 'days']
    bits = date_time.split(':')
    sec, micro = bits[-1].split('.')
    micro = ''.join(['0.', micro])
    bits = bits[:-1] + [float(b) * x for b, x in zip([sec, micro], [1, 1e6])]
    params_dict = {l : float(x) for l, x in zip(labels, reversed(bits))}
    return timedelta(**params_dict)


def update_dot_times(dot):
    """
    Read list of dot dictionaries with keys: type, name, params

    Returns a list of dict objects with three additional keys:
    - date (used to sort),
    - tstart (=tstop of a dwell in case of MANVR.ATS),
    - tstop (=tstart of a next maneuver in case of MANVR.ATS)

    Example
    =======

    >>> from parse_cm import read_dot_as_list
    >>> dot_lines = [' ATS,OBSID,ID=50411,TIME=2017:007:23:45:36.709                         OP06040001']
    >>> dot = read_dot_as_list(dot_lines)
    >>> dot
    [{'name': 'OBSID',
      'params': {'ID': 50411, 'TIME': '2017:007:23:45:36.709'},
      'type': 'ATS'}]
    >>> from flicker import update_dot_times
    >>> update_dot_times(dot)
    [{'date': '2017:007:23:45:36.709',
      'tstart': '2017:007:23:45:36.709',
      'tstop': '2017:007:23:45:36.709',
      'name': 'OBSID',
      'params': {'ID': 50411, 'TIME': '2017:007:23:45:36.709'}}]

    >>> dot_lines = ['ATS,MANVR,Q1=-0.462147874359,Q2=-0.632450473022,Q3=0.228874621726,    EP06040002',
                     "MANSTART=000:00:00:00.000,CHANGE_RATE='FALSE',HW='RWA',FSS='OUT',     EP06040002",
                     "MECH_MOVE='FALSE',DURATION=000:00:25:24.750,                          EP06040002",
                     "TIME=2017:007:23:45:36.709                                            EP06040002"]
    >>> dot = read_dot_as_list(dot_lines)
    >>> dot
    [{'name': 'MANVR',
      'params': {'CHANGE_RATE': "'FALSE'",
      'DURATION': '000:00:25:24.750',
      'FSS': "'OUT'",
      'HW': "'RWA'",
      'MANSTART': '000:00:00:00.000',
      'MECH_MOVE': "'FALSE'",
      'Q1': -0.462147874359,
      'Q2': -0.632450473022,
      'Q3': 0.228874621726,
      'TIME': '2017:007:23:45:36.709'},
      'type': 'ATS'}]

    >>> update_dot_times(dot)
    [{'date': '2017:007:23:42:36.709',
      'tstart': '2017:007:23:45:46.960',
      'tstop': '2017:008:00:11:15.811',
      'name': 'MANVR',
      'params': {'CHANGE_RATE': "'FALSE'",
      'DURATION': '000:00:25:24.750',
      'FSS': "'OUT'",
      'HW': "'RWA'",
      'MANSTART': '000:00:00:00.000',
      'MECH_MOVE': "'FALSE'",
      'Q1': -0.462147874359,
      'Q2': -0.632450473022,
      'Q3': 0.228874621726,
      'TIME': '2017:007:23:45:36.709'}}]

    :dot: list of dict objects
    :returns: list of dict objects with additional time keys:
              tstart, tstop, date
    """

    delta = {'OBSID': '00:00:00.000',
             'AOACRSTD': '-00:03:00.000',  # minus!
             'MANVR_TSTART': '00:00:10.251',
             'ACQ': '00:00:06.000'}
    
    dot_out = []

    for dd in dot:

        try:
            params = dd['params']
            time = params['TIME']
            name = dd['name']
        except KeyError:
            print('Key Error, dot entry skipped: {}'.format(dd))
            continue

        if name == 'ACQ':
            date = tstart = tstop = add_delta(time, delta['ACQ'])
        elif name == 'OBSID':
            date = tstart = tstop = add_delta(time, delta['OBSID'])
        elif name == 'MANVR':
            date = add_delta(time, delta['AOACRSTD'])
            tstart = add_delta(time, params['MANSTART'], delta['MANVR_TSTART'])
            tstop = add_delta(tstart, params['DURATION'])
        else:
            # No proper delta treatment for commands other than OBSID, ACQ, MANVR
            date = tstart = tstop = time

        # skip recording 'type', here it is always ATS
        out = {'name': name,
               'date': date,
               'tstart': tstart,
               'tstop': tstop}

        out.update({'params': params})

        dot_out.append(out)

    return dot_out


def check_iterable(*args):

    out = []

    for arg in args:
        if isinstance(arg, (list, tuple, np.ndarray)):
            out.append(arg)
        else:
            out.append([arg])

    return out


def get_deltatime(tstart, tstop):
    """
    Return list of time interval durations (tstart - tstop) in seconds

    :tstart: list of DateTime objects, format 'yyyy:ddd:hh:mm:ss.sss'
    :tstop: list of DateTime objects, format 'yyyy:ddd:hh:mm:ss.sss'
    """

    deltas = []

    tstart, tstop = check_iterable(tstart, tstop)

    for t1, t2 in zip(tstart, tstop):
        try:
            times = [datetime.strptime(t1, "%Y:%j:%H:%M:%S.%f"),
                     datetime.strptime(t2, "%Y:%j:%H:%M:%S.%f")]
        except ValueError:
            return -1

        d = (times[1] - times[0]).days * 86400
        s = (times[1] - times[0]).seconds
        us = (times[1] - times[0]).microseconds / 1e6

        deltas.append(d + s + us)

    return deltas


def get_dwells(dot):
    """
    Keep only 3 AST commands: 'MANVR', 'OBSID', 'ACQ' and
    order them by 'date'.

    Cmds list keeps track of the last 4 commands.

    If everything is OK, then at the time of recording a dwell:
    cmds = ['MANVR', 'OBSID', 'ACQ', 'MANVR'] or
    cmds = ['MANVR', 'ACQ', 'OBSID', 'MANVR']

    Cases with no 'ACQ' (e.g. cmds = ['MANVR', 'OBSID', 'MANVR', ...]) or
    an extra obsid (cmds = ['MANVR', 'OBSID', 'ACQ', 'OBSID']) are filtered out.

    Example
    =======

    >>> from parse_cm import read_dot_as_list
    >>> dot_file = "/data/mpcrit1/mplogs/2017/JAN0917/scheduled_b/md007_2305.dot"
    >>> dot = read_dot_as_list(dot_file)
    >>> import flicker
    >>> dot = flicker.update_dot_times(dot)
    >>> dwells = flicker.get_dwells(dot)
    >>> dwell = dwells[6]
    >>> dwell.keys()
    dict_keys(['tstop', 'duration', 'strcat', 'q1', 'q2', 'tstart', 'obsid', 'q3'])   

    :content: list of dictionaries containing parsed dot
    :returns: list of dwell dictionaries
    """

    dot = update_dot_times(dot)

    dot = Table(dot)
    ok = np.array ([d in ['OBSID', 'ACQ', 'MANVR'] for d in dot['name']], dtype=bool)
    dot = dot[ok]
    dot.sort('date')

    dwells = []
    dwell = {'obsid': 0, 'tstart': '', 'tstop': '',
             'duration': None, 'strcat': {},
             'q1': None, 'q2': None, 'q3': None}

    cmds = []

    for dd in dot:

        # filter out weird cases
        
        cmds.append(dd['name'])

        params = dd['params']
        
        counter = collections.Counter(cmds)
 
        if len(cmds) == 3:
            # Case of no ACQ: cmds = ['MANVR', 'OBSID', 'MANVR']
            if counter['MANVR'] == 2 and counter['OBSID'] == 1:
                print("Skip OBSID = {}".format(dwell['obsid']))
                dwell['tstart'] = dd['tstop']
                cmds = ['MANVR']
                continue
            elif len(cmds) == len(set(cmds)):
                pass
            else:
                raise ValueError("Something weird")
        elif len(cmds) > 3:
            # case of extra OBSID that should be skipped: cmds = ['MANVR', 'OBSID', 'ACQ', 'OBSID']
            if counter['OBSID'] == 2 and counter['MANVR'] == 1 and counter['ACQ'] == 1:
                # watch out: OBSID may not be the most recent entry?
                print("Skip OBSID = {}".format(params['ID']))
                cmds = ['MANVR', 'ACQ', 'OBSID']
                continue
            elif counter['OBSID'] == 1 and counter['MANVR'] == 2 and counter['ACQ'] == 1:
                # Everything is OK, update cmds, and proceed to dwell recording
                cmds = ['MANVR']
            else:
                raise ValueError("Something weird")

        # update dwell dictionary and record the dwell if 'MANVR'
            
        if dd['name'] == 'OBSID':
            dwell['obsid'] = params['ID']
        elif dd['name'] == 'ACQ':
            dwell.update({'strcat': params})
        elif dd['name'] == 'MANVR':
            dwell['tstop'] = dd['tstart']
            duration = get_deltatime(dwell['tstart'], dwell['tstop'])
            dwell['duration'] = duration[0] if hasattr(duration, '__iter__') else 0

            dwells.append(copy.copy(dwell))

            # start a new dwell record
            dwell['tstart'] = dd['tstop']
            dwell['q1'] = params['Q1']
            dwell['q2'] = params['Q2']
            dwell['q3'] = params['Q3']

    return dwells


def identify_stars(obsid, strcat, quat):
    """
    Get number of allowed monitor windows for a given star catalog.

    :obsid: observation id
    :strcat: Star catalog (see ACQ.ATS)
    :quat: quaternion

    :returns: list of dictionaries with keys: agasc_id, mag, color, strcat_no, aqflg, dist
    """

    date = strcat['TIME']
    nument = strcat['NUMENT']
    radius = (100. * u.arcsec).to('deg').value

    stars = []

    # AQFLG: 0 ACQ, 1 GUI, 2 BOT, 3 FID, 4 MON
    for i in range(nument):
        aqflg = strcat['AQFLG{}'.format(i + 1)]
        if aqflg in [1, 2]: # GUI or BOT

            fnt = strcat['AQMFNT{}'.format(i + 1)]
            brt = strcat['AQMBRT{}'.format(i + 1)]
            yag = (strcat['AQY{}'.format(i + 1)] * u.radian).to('degree').value
            zag = (strcat['AQZ{}'.format(i + 1)] * u.radian).to('degree').value

            ra, dec = quatutil.yagzag2radec(yag, zag, quat)
            cat = agasc.agasc.get_agasc_cone(ra, dec, radius, date)

            if len(cat) == 0:
                msg = 'OBSID = {}: star = {} not found in agasc.'.format(obsid, i + 1)
                raise ValueError(msg)

            for cc in cat:
                mag = cc['MAG_ACA']
                if mag < fnt and mag > brt:

                    mean_mag = np.mean([fnt, brt])
                    if np.abs(mag - mean_mag) > 1e-4:
                        msg = ' '.join(['OBSID = {}, star = {}:'.format(obsid, i + 1),
                                        'AGASC_ID = {}, MAG_ACA = {:.4f},'.format(cc['AGASC_ID'], mag),
                                        'mean(FNT, BRT) = {:.4f}'.format(mean_mag)])
                        print(msg)
                    else:
                      stars.append({'agasc_id':cc['AGASC_ID'],
                                    'mag':mag,
                                    'color':cc['COLOR1']})

    counter = collections.Counter([strcat['AQFLG{}'.format(i + 1)] for i in range(nument)])

    if len(stars) != counter[1] + counter[2]:
        print(stars)
        msg = ' '.join(['\nOBSID = {}:'.format(obsid),
                        'not all stars identified, or multiple agasc matches.'])
        raise ValueError(msg)

    return stars


def monitor_window_stats(dwells, dur_thres = 7000, t_ccd = -11, prob_thres = 0.1):
    """
    Get number of allowed monitor windows for each ER dwell with
    duration longer than dur_thres in sec.

    :dwells: list of dictionaries with dwells parsed from the DOT
    :dur_thres: duration threshold in sec for an ER dwell
    :t_ccd: temperature of the CCD in C
    :prob_thres: acquisition probability threshold to detect exactly n stars
                 (n=6 for 1 monitor window, n=7 for 2 monitor windows)
    :returns: list of dictionaries with dwell obsid and duration info
              and star acquisition and monitor window stats.
              This could be simplified so that only the numer of allowed
              monitor windows is returned. Current output is for testing
              purposes.
    """
    stats = []

    for dwell in dwells:
        obsid = dwell['obsid']
        if obsid < 60000 and obsid > 50000 and dwell['duration'] > dur_thres:
            q1 = dwell['q1']
            q2 = dwell['q2']
            q3 = dwell['q3']
            q4 = np.sqrt(1. - q1**2 - q2**2 - q3**2)
            quat = Quat(normalize([q1, q2, q3, q4]))
            strcat = dwell['strcat']
            date = strcat['TIME']
            duration = dwell['duration']

            stars = identify_stars(obsid, strcat, quat)

            t = Table(stars)
            t['probs'] = star_probs.acq_success_prob(date, t_ccd, t['mag'], t['color'])
            t.sort('probs')

            dwell_stats = {'obsid': obsid,
                           'duration': duration,
                           'date': date,
                           'quat': quat,
                           'mon_windows_1': False,
                           'mon_windows_2': False,
                           'num_mon_windows': 0,
                           'stars_dropped_1': [],
                           'stars_dropped_2': []}

            for i in [6, 7]:
                p, c = star_probs.prob_n_acq(t['probs'][:i])
                if p[-1] > prob_thres:
                    num_mon_windows = 8 - i
                    dwell_stats['mon_windows_{}'.format(num_mon_windows)] = True
                    if num_mon_windows > dwell_stats['num_mon_windows']:
                        dwell_stats['num_mon_windows'] = num_mon_windows
                    if num_mon_windows > 0:
                        # improve this: star probs could be equal to each other,
                        # and then it doesnt matter which star is dropped
                        stars_dropped = [id for id in t['agasc_id'][-num_mon_windows:]]
                        dwell_stats['stars_dropped_{}'.format(num_mon_windows)] = stars_dropped

                dwell_stats['prob_{}'.format(i)] = format(p[-1], '.3f')

            stats.append(dwell_stats)

    return stats


def get_spoiler_stars(rows, cols, vals, stars={}, rim=20):
    """
    """
    for r, c, val in zip(rows, cols, vals):
        r1 = max(r - rim, -512)
        r2 = min(r + rim, 511)
        c1 = max(c - rim, -512)
        c2 = min(c + rim, 511)
        for rr in np.arange(r1, r2):
            for cc in np.arange(c1, c2):
                if (rr, cc) not in stars.keys():
                    stars[(rr, cc)] = val
    return stars


def schedule_predefined(locations, date, quat, num_allowed, mag_thres=13, radius=25):
    """
    :locations: astropy Table with predefined yag, zag, row0, col0 coordinates,
                and number of times the location has been scheduled in the past
    :date: date
    :quat: pointing quaternion of this dwell
    :num_allowed: allowed number of monitor windows for this dwell
    :mag_thres: faintest magnitude of a spoiler star. Default = 13 mag
    :radius: radius of search for spoiler stars. Default = 25 arcsec
    """

    radius = (radius * u.arcsec).to('deg').value

    num_scheduled = 0

    idx_list = []

    for i, row in enumerate(locations):

        if num_scheduled < num_allowed and row['scheduled'] != 1:

            r = row['row']
            c = row['col']
            yag, zag = transform.pixels_to_yagzag(r, c) # yag, zag in arcsec
            ra, dec = quatutil.yagzag2radec(yag / 3600., zag / 3600., quat)
            cat = agasc.agasc.get_agasc_cone(ra, dec, radius, date)

            if all(cat['MAG_ACA']) > mag_thres or len(cat) == 0:
                msg = ' '.join(['Schedule {} monitor window at:\n'.format(num_scheduled + 1),
                                '    (yag, zag) = ({:.0f}, {:.0f}) arcsec\n'.format(yag, zag),
                                '    (row, col) = ({}, {})'.format(r, c)])
                print(msg)
                num_scheduled = num_scheduled + 1
                idx_list.append(i)
            """
            else:
                msg = ' '.join(['Spoiler star(s) found at:\n',
                                '    (yag, zag) = ({:.0f}, {:.0f}) arcsec\n'.format(yag, zag),
                                '    (row, col) = ({}, {})'.format(r, c)])
                print(msg)
                ok = cat['MAG_ACA'] <= mag_thres
                print(cat['AGASC_ID'][ok])
            """

    return idx_list


def schedule_monitor_windows(t, predefined_locs=False, mag_thres=13, radius=25):
    """
    :t: astropy Table
    :predefined_locs: if True, try the predefined CCD locations first,
                      default=False.
    """
    locations = Table.read(PREDEFINED, delimiter=' ', format='ascii')

    for row in t:

        obsid = row['obsid']
        dur = row['duration']
        num_allowed = row['num_mon_windows']

        msg = ' '.join(['\nObsID = {}, duration = {:.0f} sec:'.format(obsid, dur),
                        '{} monitor windows allowed'.format(num_allowed)])
        print(msg)

        date = row['date']
        quat = row['quat']

        num_scheduled = 0

        stars = {}

        # If flag set, try predefined locations first
        if predefined_locs:

            print("Predefined locations:")

            kwargs = {'mag_thres': mag_thres, 'radius': radius}

            if not all(locations['scheduled']) == 1:
                idx_list = schedule_predefined(locations, date, quat, num_allowed, **kwargs)
                num_scheduled = len(idx_list)
                # Update status of predefined locations
                for idx in idx_list:
                    locations['scheduled'][idx] = 1

        # All allowed predefined locations are scheduled, or predefined_locs=False,
        # Add random locations if needed
        if num_scheduled < num_allowed:
 
            print("Random locations:")

            ra, dec = quatutil.yagzag2radec(0, 0, quat)
            cat = agasc.agasc.get_agasc_cone(ra, dec, 1.5, date)

            # Filter to pick up bright spoiler stars
            ok = np.array(cat['MAG_ACA'] < mag_thres, dtype=bool)
            cat = cat[ok]

            yags, zags = quatutil.radec2yagzag(cat['RA_PMCORR'], cat['DEC_PMCORR'], quat)
            rows, cols = transform.yagzag_to_pixels(yags * 3600., zags * 3600, allow_bad=True)
            
            # Identify spoiler stars that fit on ccd
            ok = (rows > -512.5) * (rows < 511.5) * (cols > -512.5) * (cols < 511.5)
               
            rows = np.array(np.round(rows[ok]), dtype=int)
            cols = np.array(np.round(cols[ok]), dtype=int)

            # Collect spoiler stars
            vals = np.ones(len(rows))
            stars = get_spoiler_stars(rows, cols, vals)

            # Now dict stars contains keys that represent (row, col) of
            # spoiler star centers and a 10px rim around each spoiler star.

            while num_scheduled < num_allowed:
                # Draw a random location on ccd, avoid edges:
                r, c = np.random.randint(-504, 505, 2)

                # Check if the location was previously scheduled - TBD

                # Check if the location is free of stars
                if (r, c) not in stars:
                    yag, zag = transform.pixels_to_yagzag(r, c) # yag, zag in arcsec
                    msg = ' '.join(['Schedule {} monitor window at:\n'.format(num_scheduled + 1),
                                    '    (yag, zag) = ({:.0f}, {:.0f}) arcsec\n'.format(yag, zag),
                                    '    (row, col) = ({}, {})'.format(r, c)])
                    print(msg)
                    num_scheduled = num_scheduled + 1
                    stars = get_spoiler_stars([r], [c], [5], stars, rim=4)

    # Update the PREDEFINED file - TBD

    return stars


def scheduler(dot, dur_thres=7000, t_ccd=-11, prob_thres=0.1, mag_thres=13, radius=25, \
              predefined_locs=False):
    """
    Schedule monitor windows at predefined and/or random locations.

    :dot: parsed dot file, list of dictionaries
    :dur_thres: minimum duration of dwell in seconds, default=7000 sec
    :t_ccd: CCD temperature in degrees C, default=-11 C
    :prob_thres: minimum probability of detecting n stars
    :mag_thres: faintest magnitude of spoiler stars, default=13 mag
    :radius: radius of search for spoiler stars in arcsec, default=25 arcsec
    :predefined_locs: if True, try the predefined CCD locations first,
                      default=False. Predefined locations are read from a file
                      predefined_locations.txt (defined with PREDEFINED). The
                      file has 3 columns: row, col, scheduled;
                      scheduled=0: location has not been scheduled yet, try to
                      schedule it;
                      scheduled=1: location has already been scheduled, skip it.
                      Note: Currently only the first two predefined locations have
                      scheduled=0. The second predefined location coincides with
                      a spoiler star for ObsID=50406 used in this test.

    Example:
    =======

    >> from monwin import scheduler_test
    >> from parce_cm import read_dot_as_list
    >> dot = read_dot_as_list("/data/mpcrit1/mplogs/2017/JAN0917/scheduled_b/md007_2305.dot")
    >> stars = scheduler(dot, predefined_locs=True)
    Skip OBSID = 50405
    Skip OBSID = 50398
    Skip OBSID = 50396
    Skip OBSID = 50394
    Skip OBSID = 50392
    Skip OBSID = 50390

    ObsID = 50406, duration = 11594 sec: 2 monitor windows allowed
    Predefined locations:
    Schedule 1 monitor window at:
        (yag, zag) = (-2017, -2522) arcsec
        (row, col) = (413, -506)
    Random locations:
    Schedule 2 monitor window at:
        (yag, zag) = (49, 2286) arcsec
        (row, col) = (-3, 465)

    ObsID = 50402, duration = 8972 sec: 2 monitor windows allowed
    Predefined locations:
    Schedule 1 monitor window at:
        (yag, zag) = (-2196, 2343) arcsec
        (row, col) = (451, 478)
    Random locations:
    Schedule 2 monitor window at:
        (yag, zag) = (-2259, 1970) arcsec
        (row, col) = (463, 402)

    ObsID = 50401, duration = 13079 sec: 2 monitor windows allowed
    Predefined locations:
    Random locations:
    Schedule 1 monitor window at:
        (yag, zag) = (-1704, -500) arcsec
        (row, col) = (348, -96)
    Schedule 2 monitor window at:
        (yag, zag) = (311, -1584) arcsec
        (row, col) = (-57, -313)

    ObsID = 50381, duration = 15088 sec: 2 monitor windows allowed
    Predefined locations:
    Random locations:
    Schedule 1 monitor window at:
        (yag, zag) = (1608, -2100) arcsec
        (row, col) = (-319, -418)
    Schedule 2 monitor window at:
        (yag, zag) = (547, -1053) arcsec
        (row, col) = (-104, -206)
    """

    dwells = get_dwells(dot)
    kwargs = {'dur_thres': dur_thres, 't_ccd': t_ccd, 'prob_thres': prob_thres}
    stats = monitor_window_stats(dwells, **kwargs)

    t = Table(stats)

    kwargs = {'mag_thres': mag_thres, 'radius': radius}
    stars = schedule_monitor_windows(t, predefined_locs, **kwargs)

    return stars


# Tests

def dwells_test(dot, dur_thres = 7000, t_ccd = -11, prob_thres = 0.1):
    """
    Compare dwell start and stop times between kadi and code output.
    Compute difference between tstarts and tstops recorded in kadi and
    found by this code.

    Example:
    =======

    >> from monwin import dwells_test
    >> from parce_cm import read_dot_as_list
    >> dot = read_dot_as_list("/data/mpcrit1/mplogs/2017/JAN0917/scheduled_b/md007_2305.dot")
    >> dwells_test(dot)
    Skip OBSID = 50405
    Skip OBSID = 50398
    Skip OBSID = 50396
    Skip OBSID = 50394
    Skip OBSID = 50392
    Skip OBSID = 50390
    OBSID = 50411
    kadi: tstart = 2017:008:00:12:57.527, tstop = 2017:008:01:05:18.127, duration = 3141 sec
    code: tstart = 2017:008:00:11:11.710, tstop = 2017:008:01:05:29.308, duration = 3258 sec
    Delta tstart = -105.817, Delta tstop = 11.181

    OBSID = 50410
    kadi: tstart = 2017:008:01:45:00.227, tstop = 2017:008:02:37:14.677, duration = 3134 sec
    code: tstart = 2017:008:01:42:56.521, tstop = 2017:008:02:37:26.605, duration = 3270 sec
    Delta tstart = -123.706, Delta tstop = 11.928
    ...
    ...
    ...
    """

    dwells = get_dwells(dot)
    stats = monitor_window_stats(dwells, dur_thres = 7000, t_ccd = -11, prob_thres = 0.1)

    from kadi import events
    from Chandra.Time import DateTime

    t = Table(dwells)

    for i, obsid in enumerate(t['obsid']):
        if not obsid > 0:
            continue

        d = events.dwells.filter(obsid=obsid)[0]

        times = [[DateTime(d.tstart).date, DateTime(d.tstop).date],
                 [t['tstart'][i], t['tstop'][i]]]

        deltas = get_deltatime(times[0], times[1])

        msg = ("OBSID = {}\n".format(obsid) +
               "kadi: tstart = {}, tstop = {}, duration = {:.0f} sec\n".format(times[0][0],
                                                                               times[0][1],
                                                                               d.tstop - d.tstart) +
               "code: tstart = {}, tstop = {}, duration = {:.0f} sec\n".format(times[1][0],
                                                                               times[1][1],
                                                                               t['duration'][i]) +
               "Delta tstart = {:.3f}, Delta tstop = {:.3f}\n".format(deltas[0], deltas[1]))
        print(msg)

    return


def stars_test(dwells, dur_thres = 7000):
    """
    Check that the appropriate stars are identified in agasc.
    For a given obsid, star id's and mags can be compared with those in starcheck.

    Example:
    =======

    >> from monwin import stars_test, get_dwells
    >> from parce_cm import read_dot_as_list
    >> dot = read_dot_as_list("/data/mpcrit1/mplogs/2017/JAN0917/scheduled_b/md007_2305.dot")
    >> dwells = get_dwells(dot)
    >> stars_test(dwells)
    Obsid = 0, duration = 0 sec, skipped
    Obsid = 50411, duration = 3258 sec, skipped
    Obsid = 50410, duration = 3270 sec, skipped
    Obsid = 50409, duration = 3248 sec, skipped
    Obsid = 50408, duration = 1242 sec, skipped
    Obsid = 50407, duration = 3243 sec, skipped
    Obsid = 50406, duration = 11594 sec
     agasc_id  color     mag  
    --------- -------- -------
    389949248  0.95285 7.53174
    389953880  0.12495 7.79542
    389954848 0.113901 8.71616
    389954920  0.54995 9.34569
    390865160   1.0302 8.71366
    390867952      1.5 8.68183
    390339464  0.48875 9.34993
    391254944   0.2176 7.91407
    Obsid = 18048, duration = 71020 sec, skipped
    ...
    ...
    ...
    """

    for dwell in dwells:
        obsid = dwell['obsid']
        duration = dwell['duration']
        if obsid < 60000 and obsid > 50000 and duration > dur_thres:
            strcat = dwell['strcat']
            q1 = dwell['q1']
            q2 = dwell['q2']
            q3 = dwell['q3']
            q4 = np.sqrt(1. - q1**2 - q2**2 - q3**2)
            quat = Quat(normalize([q1, q2, q3, q4]))
            stars = identify_stars(obsid, strcat, quat)
            print('Obsid = {}, duration = {:.0f} sec'.format(obsid, duration))
            print(Table(stars))
        else:
            print('Obsid = {}, duration = {:.0f} sec, skipped'.format(obsid, duration))

    return


def scheduler_test(dot, num=10, dur_thres=7000, t_ccd=-11, prob_thres=0.1, \
                mag_thres=13, radius=25, predefined_locs=False):
    """
    Test method that schedules monitor windows at random locations.
    Test on the first dwell that passes monitor_window_stats check,
    schedule num random monitor windows (default num = 10).

    The test outputs a plot with spoiler star centroids and a region around
    them with size defined with variable 'rim' (see get_spoiler_stars) marked
    in red, and scheduled monitor windows (8x8) marked in white.

    :dot: parsed dot file, list of dictionaries
    :num: numer of monitor windows to be scheduled, overwrites the real numer
    :dur_thres: minimum duration of dwell in seconds, default=7000 sec
    :t_ccd: CCD temperature in degrees C, default=-11 C
    :prob_thres: minimum probability of detecting n stars
    :mag_thres: faintest magnitude of spoiler stars, default=13 mag
    :radius: radius of search for spoiler stars in arcsec, default=25 arcsec
    :predefined_locs: if True, try the predefined CCD locations first,
                      default=False. Predefined locations are read from a file
                      predefined_locations.txt (defined with PREDEFINED). The
                      file has 3 columns: row, col, scheduled;
                      scheduled=0: location has not been scheduled yet, try to
                      schedule it;
                      scheduled=1: location has already been scheduled, skip it.
                      Note: Currently only the first two predefined locations have
                      scheduled=0. The second predefined location coincides with
                      a spoiler star for ObsID=50406 used in this test.

    Example:
    =======

    >> from monwin import scheduler_test
    >> from parce_cm import read_dot_as_list
    >> dot = read_dot_as_list("/data/mpcrit1/mplogs/2017/JAN0917/scheduled_b/md007_2305.dot")
    >> scheduler_test(dot)
    Skip OBSID = 50405
    Skip OBSID = 50398
    Skip OBSID = 50396
    Skip OBSID = 50394
    Skip OBSID = 50392
    Skip OBSID = 50390

    ObsID = 50406, duration = 11594 sec: 10 monitor windows allowed
    Random locations:
    Schedule 1 monitor window at:
        (yag, zag) = (-1085, 715) arcsec
        (row, col) = (224, 148)
    Schedule 2 monitor window at:
        (yag, zag) = (1172, -1440) arcsec
        (row, col) = (-230, -284)
    Schedule 3 monitor window at:
        (yag, zag) = (-271, 1402) arcsec
        (row, col) = (61, 286)
    ...
    ...
    ...

    >> scheduler_test(dot, predefined_locs=True)
    Skip OBSID = 50405
    Skip OBSID = 50398
    Skip OBSID = 50396
    Skip OBSID = 50394
    Skip OBSID = 50392
    Skip OBSID = 50390

    ObsID = 50406, duration = 11594 sec: 10 monitor windows allowed
    Predefined locations:
    Schedule 1 monitor window at:
        (yag, zag) = (-2017, -2522) arcsec
        (row, col) = (413, -506)
    Random locations:
    Schedule 2 monitor window at:
        (yag, zag) = (-1292, -436) arcsec
        (row, col) = (265, -83)
    Schedule 3 monitor window at:
        (yag, zag) = (649, -1782) arcsec
        (row, col) = (-125, -353)
    ...
    ...
    ...

    """

    dwells = get_dwells(dot)
    kwargs = {'dur_thres': dur_thres, 't_ccd': t_ccd, 'prob_thres': prob_thres}
    stats = monitor_window_stats(dwells, **kwargs)

    t = Table(stats)
    # For testing, alter the number of allowed monitor windows
    t[0]['num_mon_windows'] = num

    kwargs = {'predefined_locs': predefined_locs, 'mag_thres': mag_thres, 'radius': radius}
    stars = schedule_monitor_windows(t[:1], **kwargs)

    ccd = np.zeros((1024, 1024))
    for r in np.arange(-512, 511):
        for c in np.arange(-512, 511):
            if (r, c) in stars.keys():
                ccd[r + 512, c + 512] = stars[(r, c)]

    plt.figure()
    plt.imshow(ccd, origin='lower', cmap='hot', interpolation=None)

    return
