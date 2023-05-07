"""
Prepare and save train, test, dev data for a multi-label classification model predicting domains in a sentence.

Regarding the data split, there are two options:
(1) the data is split into train (80%), test (10%), dev (10%)
(2) if existing dev and test sets are given, only a train set is created (which excludes the notes in dev and test)

The train set can optionally be altered in two ways:
(1) background/target sentences that don't have any domain labels are excluded
(2) positive examples from the pilot data are added (note that only 4 domains are annotated: BER, FAC, INS, STM)

The script can be customized with the following parameters:
    --datapath: data dir with parsed annotations in pkl format
    --split: slpit the data to train/test/dev
    --pilot: add pilot data to the train set
    --bck: exclude background/target sentences from the train set
    --test: exisiting test data (pkl)
    --dev: existing dev data (pkl)

To change the default values of a parameter, pass it in the command line, e.g.:

$ python data_prep_domains.py --pilot --bck
"""


import argparse
import spacy
import pandas as pd

import sys
sys.path.insert(0, '../..')
# from utils.config import PATHS
from utils.data_process import concat_annotated, drop_disregard, fix_week_14, pad_sen_id, anonymize, data_split_groups


def exclude_bck(df):
    """
    Exclude background/target sentences that don't have any domain labels.
    """
    print(f"Before excl: {len(df)=}")
    crit1 = "(background_sent or target_sent)"
    crit2 = "labels.astype('str') == '[0, 0, 0, 0, 0, 0, 0, 0, 0]'"
    to_excl = df.query(f"{crit1} and {crit2}")
    df = df.loc[~df.index.isin(to_excl.index)]
    print(f"After excl: {len(df)=}")
    return df


def main(
    datapath,
    split_data,
    add_pilot,
    excl_bck,
    test,
    dev,
    domains=['ADM', 'ATT', 'BER', 'ENR', 'ETN', 'FAC', 'INS', 'MBW', 'STM'],
):
    """
    Prepare and save train, test, dev data for a multi-label classification model predicting domains in a sentence.

    Regarding the data split, there are two options:
    (1) the data is split into train (80%), test (10%), dev (10%)
    (2) if existing dev and test sets are given, only a train set is created (which excludes the notes in dev and test)

    The train set can optionally be altered in two ways:
    (1) background/target sentences that don't have any domain labels are excluded
    (2) positive examples from the pilot data are added (note that only 4 domains are annotated: BER, FAC, INS, STM)

    Parameters
    ----------
    datapath: Path
        path to directory with parsed annotations in pkl format
    split_data: bool
        if True, slpit the data to train/test/dev
    add_pilot: bool
        if True, add pilot data to the train set
    excl_bck: bool
        if True, exclude background/target sentences from the train set
    test: Path or None
        path exisiting test data (pkl)
    dev: Path or None
        path to existing dev data (pkl)

    Returns
    -------
    None
    """

    #### PROCESS ANNOTATIONS ####

    other = ['target', 'background', 'plus']

    datapath = './../data_process_from_inception/data/CORRECT_parse_annotations_output.pkl' 
    # load and pre-process data
    print(f"Pre-processing data in {datapath}...")
    df = pd.read_pickle(datapath)
    
    # sentence-level pre-processing
    df = df.assign(
        background_sent = lambda df: df.groupby('sen_id').background.transform('any'),
        target_sent = lambda df: df.groupby('sen_id').target.transform('any'),
        pad_sen_id = df.sen_id.apply(pad_sen_id)
    )

    # fill NA
    df[domains + other] = df[domains + other].fillna(False)
    df[['label', 'relation']] = df[['label', 'relation']].fillna('_')
    df['token'] = df['token'].fillna('')

    #### SENTENCE-LEVEL DF ####
    
    print("Creating sentence-level df...")
    info_cols = ['pad_sen_id', 'NotitieID', 'background_sent', 'target_sent']
    info = df.groupby('pad_sen_id')[info_cols].first()
    text = df.groupby('pad_sen_id').token.apply(lambda s: s.str.cat(sep=' ')).rename('text_raw')
    labels = df.groupby('pad_sen_id')[domains].any().astype(int).apply(lambda s: s.to_list(), axis=1).rename('labels')
    df = pd.concat([info, text, labels], axis=1)

    # outdir to save the data
    outdir = './../data_process_from_inception/data'

    #### SPLIT DATA ####
    
    if split_data:
        # anonymize text
        print("Anonymizing text...")
        nlp = spacy.load('nl_core_news_lg')
        df = df.join(
            df.apply(
                lambda row: anonymize(row.text_raw, nlp),
                axis=1,
                result_type='expand',
            ).rename(columns={0: 'text', 1: 'len_text'})
        )       
        # split
        print("Splitting to train / dev / test...")
        train, dev, test = data_split_groups(
            df,
            'text',
            'labels',
            'NotitieID',
            0.8,
        )    
        # save
        dev.to_pickle('./../data_process_from_inception/data/CORRECT_development_set.pkl')
        dev.to_csv('./../data_process_from_inception/data/CORRECT_development_set.csv')
        test.to_pickle('./../data_process_from_inception/data/CORRECT_test_set.pkl')
        test.to_pickle('./../data_process_from_inception/data/CORRECT_test_set.csv')
        print(f"dev and test sets are saved")
    
    else:
        dev = pd.read_pickle(dev)
        test = pd.read_pickle(test)
        train = df.query("NotitieID not in @test.NotitieID and NotitieID not in @dev.NotitieID")
        # anonymize text
        print("Anonymizing text in train set...")
        nlp = spacy.load('nl_core_news_lg')
        train = train.join(
            train.apply(
                lambda row: anonymize(row.text_raw, nlp),
                axis=1,
                result_type='expand',
            ).rename(columns={0: 'text', 1: 'len_text'})
        )

    #### ADDITIONAL OPERATIONS ON TRAIN DATA ####

    
    # save
    train.to_pickle('./../data_process_from_inception/data/CORRECT_train_set.pkl')
    print(f"training set is saved")
    train.to_csv('./../data_process_from_inception/data/CORRECT_train_set.csv')
        
    if excl_bck:
        print('Excluding background/target from the train set...')
        train = exclude_bck(train)
        train_filename = train_filename + '_excl_bck'
        # save
        train.to_pickle(outdir / f'{train_filename}.pkl')
        print(f"{train_filename} set is saved to: {outdir}")

    if add_pilot:
        print('Adding pilot data to the train set...')
        train = add_pilot_data(train, datapath)
        train_filename = train_filename + '_add_pilot'
        # save
        train.to_pickle(outdir / f'{train_filename}.pkl')
        print(f"{train_filename} set is saved to: {outdir}")


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--datapath', default='./draft_output.pkl', help='must be listed as a key in /config.ini')
    argparser.add_argument('--split', dest='split_data', action='store_true')
    argparser.set_defaults(split_data=True)
    argparser.add_argument('--pilot', dest='add_pilot', action='store_true')
    argparser.set_defaults(add_pilot=False)
    argparser.add_argument('--bck', dest='excl_bck', action='store_true')
    argparser.set_defaults(excl_bck=False)
    argparser.add_argument('--test', default='clf_domains/test.pkl', help=' automatically being set to None if split_data is True')
    argparser.add_argument('--dev', default='clf_domains/dev.pkl', help=' automatically being set to None if split_data is True')
    args = argparser.parse_args()

#     datapath = PATHS.getpath(args.datapath)
#     if args.split_data:
#         test = None
#         dev = None
#     else:
#         test = datapath / args.test
#         dev = datapath / args.dev
    
    test = None
    dev = None 
    datapath = './../data_process_from_inception/data/CORRECT_parse_annotations_output.pkl' 
    
    main(
        datapath,
        args.split_data,
        args.add_pilot,
        args.excl_bck,
        test,
        dev,
    )