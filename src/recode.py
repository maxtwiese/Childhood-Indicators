import numpy as np

def recoder(df):

    # properly code in NaN values
    df.replace(to_replace = [85, 94, 97, 98, 99, 985, 994, 997, 998, 999],
        value = np.nan, inplace=True)

    # recode Environemntal Indicators
    df.BOOKED = df.BOOKED.map({2:0, 3: 1}, na_action='ignore')
    df.PDEN10 = df.PDEN10.map({1:2, 2: 1, 3: 0}, na_action='ignore')
    df.COUTYP4 = df.COUTYP4.map({1:2, 2: 1, 3: 0}, na_action='ignore')

    # recode Peer/Social Indicators
    df.YESTSCIG = df.YESTSCIG.map({1:0, 2:1, 3:2, 4:3}, na_action='ignore')
    df.YESTSMJ = df.YESTSMJ.map({1:0, 2:1, 3:2, 4:3}, na_action='ignore')
    df.YESTSALC = df.YESTSALC.map({1:0, 2:1, 3:2, 4:3}, na_action='ignore')
    df.YESTSDNK = df.YESTSDNK.map({1:0, 2:1, 3:2, 4:3}, na_action='ignore')
    df.YEYFGTSW = df.YEYFGTSW.map({1:0, 2:1, 4: 6, 5: 10}, na_action='ignore')
    df.YEYFGTGP = df.YEYFGTGP.map({1:0, 2:1, 4: 6, 5: 10}, na_action='ignore')

    # recode School Indicators
    df.YEATNDYR = df.YEYFGTGP.map({2:0}, na_action='ignore')
    df.YESCHFLT = df.YESCHFLT.map({1:3, 3:1, 4: 0}, na_action='ignore')
    df.YESCHWRK = df.YESCHWRK.map({1:3, 3:1, 4: 0}, na_action='ignore')
    df.YESCHIMP = df.YESCHIMP.map({1:3, 3:1, 4: 0}, na_action='ignore')
    df.YESCHINT = df.YESCHINT.map({1:3, 3:1, 4: 0}, na_action='ignore')
    df.YETCGJOB = df.YETCGJOB.map({1:3, 3:1, 4: 0}, na_action='ignore')
    df.YELSTGRD = df.YELSTGRD.map({1:3, 3:1, 4: 0}, na_action='ignore')

    # recode Parental Indicators
    df.YEPCHKHW = df.YEPCHKHW.map({1:3, 3:1, 4: 0}, na_action='ignore')
    df.YEPHLPHW = df.YEPHLPHW.map({1:3, 3:1, 4: 0}, na_action='ignore')
    df.YEPCHORE = df.YEPCHORE.map({1:3, 3:1, 4: 0}, na_action='ignore')
    df.YEPLMTTV = df.YEPLMTTV.map({1:3, 3:1, 4: 0}, na_action='ignore')
    df.YEPLMTSN = df.YEPLMTSN.map({1:3, 3:1, 4: 0}, na_action='ignore')
    df.YEPGDJOB = df.YEPGDJOB.map({1:3, 3:1, 4: 0}, na_action='ignore')
    df.YEPPROUD = df.YEPPROUD.map({1:3, 3:1, 4: 0}, na_action='ignore')
    df.YEYARGUP = df.YEYARGUP.map({1:0, 2:1, 4: 6, 5: 10}, na_action='ignore')
    df.YEPPKCIG = df.YEPPKCIG.map({1: 2, 2: 1, 3: 0}, na_action='ignore')
    df.YEPMJEVR = df.YEPMJEVR.map({1: 2, 2: 1, 3: 0}, na_action='ignore')
    df.YEPMJMO = df.YEPMJMO.map({1: 2, 2: 1, 3: 0}, na_action='ignore')
    df.YEPALDLY = df.YEPALDLY.map({1: 2, 2: 1, 3: 0}, na_action='ignore')
    df.YEPRTDNG = df.YEPRTDNG.map({2: 0}, na_action='ignore')

    # recode Interal Indicators
    df.YEGPKCIG = df.YEGPKCIG.map({1: 2, 2: 1, 3: 0}, na_action='ignore')
    df.YEGMJEVR = df.YEGMJEVR.map({1: 2, 2: 1, 3: 0}, na_action='ignore')
    df.YEGMJMO = df.YEGMJMO.map({1: 2, 2: 1, 3: 0}, na_action='ignore')
    df.YEGALDLY = df.YEGALDLY.map({1: 2, 2: 1, 3: 0}, na_action='ignore')
    df.YEFPKCIG = df.YEFPKCIG.map({1: 2, 2: 1, 3: 0}, na_action='ignore')
    df.YEFMJEVR = df.YEFMJEVR.map({1: 2, 2: 1, 3: 0}, na_action='ignore')
    df.YEFMJMO = df.YEFMJMO.map({1: 2, 2: 1, 3: 0}, na_action='ignore')
    df.YEFALDLY = df.YEFALDLY.map({1: 2, 2: 1, 3: 0}, na_action='ignore')
    df.YERLGSVC = df.YERLGSVC.map({1:0, 2: 1, 4: 6, 5: 10}, na_action='ignore')
    df.YERLGIMP = df.YERLGIMP.map({1:0, 2: 1, 3: 2, 4: 3}, na_action='ignore')
    df.YERLDCSN = df.YERLDCSN.map({1:0, 2: 1, 3: 2, 4: 3}, na_action='ignore')
    df.YERLFRND = df.YERLFRND.map({1:0, 2: 1, 3: 2, 4: 3}, na_action='ignore')
    df.YODPREV = df.YODPREV.map({2: 0}, na_action='ignore')
    df.YODSCEV = df.YODSCEV.map({2: 0}, na_action='ignore')
    df.YOLOSEV = df.YOLOSEV.map({2: 0}, na_action='ignore')
    df.YOWRHRS = df.YOWRHRS.map({4: 0}, na_action='ignore')
    df.YOWRHRS = df.YOWRHRS.fillna(0)       #logical NaN -> 0
    df.YOWRDST = df.YOWRDST.fillna(0)       #logical NaN -> 0
    df.YOWRCHR = df.YOWRCHR.map({1: 3, 3: 1, 4: 0}, na_action='ignore')
    df.YOWRCHR = df.YOWRCHR.fillna(0)       #logical NaN -> 0
    df.YOWRIMP = df.YOWRIMP.map({1: 3, 3: 1, 4: 0}, na_action='ignore')
    df.YOWRIMP = df.YOWRIMP.fillna(0)       #logical NaN -> 0
    df.YODPPROB = df.YODPPROB.map({2: 0}, na_action='ignore')
    df.HEALTH = df.HEALTH.map({1: 4, 2: 3, 3: 2, 4: 1, 5: 0}, na_action='ignore')
    df.DIFFTHINK = df.DIFFTHINK.map({2: 0}, na_action='ignore')
    df.DIFFTHINK = df.DIFFTHINK.fillna(0)       #logical NaN -> 0

    # recode Substance Specific Education Indicators
    df.YEPRBSLV = df.YEPRBSLV.map({2: 0}, na_action='ignore')
    df.YEVIOPRV = df.YEVIOPRV.map({2: 0}, na_action='ignore')
    df.YEDGPRGP = df.YEDGPRGP.map({2: 0}, na_action='ignore')
    df.YEDECLAS = df.YEDECLAS.map({2: 0}, na_action='ignore')
    df.YEDERGLR = df.YEDERGLR.map({2: 0}, na_action='ignore')
    df.YEDESPCL = df.YEDESPCL.map({2: 0}, na_action='ignore')
    df.YEPVNTYR = df.YEPVNTYR.map({2: 0}, na_action='ignore')

    # build targets and drop diminished features
    df.loc[ (df.CIGEVER == 1) | (df.SMKLSSEVR == 1) | (df.CIGAREVR == 1) |
        (df.PIPEVER == 1), 'TOBACCO'] = 1

    df.loc[ (df.CIGEVER != 1) & (df.SMKLSSEVR != 1) & (df.CIGAREVR != 1) &
        (df.PIPEVER != 1), 'TOBACCO'] = 0

    df.loc[df.ALCEVER == 1, 'ALCOHOL'] = 1
    df.loc[df.ALCEVER != 1, 'ALCOHOL'] = 0

    df.loc[df.MJEVER == 1, 'CANNABIS'] = 1
    df.loc[df.MJEVER != 1, 'CANNABIS'] = 0

    df.loc[ (df.COCEVER == 1) | (df.CRKEVER == 1) | (df.HEREVER == 1) |
        (df.HALLUCEVR == 1) | (df.INHALEVER == 1) | (df.METHAMEVR == 1) |
        (df.PNRANYLIF == 1) | (df.TRQANYLIF == 1) | (df.STMANYLIF == 1) |
        (df.SEDANYLIF == 1) | (df.PNRNMLIF == 1) | (df.COLDMEDS == 1),
        'FURTHER'] = 1

    df.loc[ (df.COCEVER != 1) & (df.CRKEVER != 1) & (df.HEREVER != 1) &
        (df.HALLUCEVR != 1) & (df.INHALEVER != 1) & (df.METHAMEVR != 1) &
        (df.PNRANYLIF != 1) & (df.TRQANYLIF != 1) & (df.STMANYLIF != 1) &
        (df.SEDANYLIF != 1) & (df.PNRNMLIF != 1) & (df.COLDMEDS != 1),
        'FURTHER'] = 0

    df.drop(columns = ['CIGEVER', 'SMKLSSEVR', 'CIGAREVR', 'PIPEVER',
        'ALCEVER', 'MJEVER', 'COCEVER', 'CRKEVER', 'HEREVER', 'HALLUCEVR',
        'INHALEVER', 'METHAMEVR', 'PNRANYLIF', 'TRQANYLIF', 'STMANYLIF',
        'SEDANYLIF', 'PNRNMLIF', 'COLDMEDS'], inplace = True)
    return df
