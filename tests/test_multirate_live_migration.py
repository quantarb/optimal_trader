from datetime import datetime, timedelta
from pathlib import Path
import json

import pandas as pd
import pytest

from app import trading_app_v2_runtime as runtime


def test_latest_warehouse_scores_use_existing_hits_policy(tmp_path):
    p=tmp_path/'scores.parquet'
    pd.DataFrame({'symbol':['A','B'],'date':['2026-06-02']*2,'asset_class':['equity']*2,
        'hits_long_return_hub':[10.,20.], 'hits_long_return_authority':[9.,8.],
        'hits_short_return_hub':[20.,10.], 'hits_short_return_authority':[1.,2.],
        'hits_long_speed_hub':[3.,4.], 'hits_long_speed_authority':[5.,6.],
        'hits_short_speed_hub':[7.,8.], 'hits_short_speed_authority':[9.,10.],
        'government_is_buy':[.8,.2], 'government_is_sell':[.2,.8],
        'insider_is_buy':[.7,.3], 'insider_is_sell':[.3,.7],
        'oracle_is_buy':[1.,0.], 'oracle_is_short':[0.,1.]}).to_parquet(p)
    scores=runtime.load_multirate_strategy_scores(p)
    assert scores['long_score'].tolist()==[.5,1.]
    assert scores['short_score'].tolist()==[1.,.5]
    assert scores['long_exit_score'].tolist()==[1.,.5]
    assert scores['hits_long_return_hub'].tolist()==[10.,20.]
    assert scores['hits_short_speed_authority'].tolist()==[9.,10.]
    assert scores['government_is_buy'].tolist()==[.8,.2]
    assert scores['insider_is_sell'].tolist()==[.3,.7]
    assert scores['strategy_source'].tolist()==['warehouse_multirate']*2


def test_latest_leaderboard_includes_hits_and_trade_event_heads():
    row = {'symbol':'A','date':'2026-06-02','strategy_source':'warehouse_multirate',
           'long_score':.9,'short_score':.1,
           **{head:index/100 for index,head in enumerate(runtime.MULTIRATE_LEADERBOARD_HEADS,1)}}
    board=runtime.build_latest_equity_leaderboard(pd.DataFrame([row]),top_k=20,price_map={'A':100.})
    assert all(head in board.columns for head in runtime.MULTIRATE_LEADERBOARD_HEADS)
    assert board.loc[0,'hits_long_return_hub']==pytest.approx(.01)
    assert board.loc[0,'government_is_buy']==pytest.approx(.09)
    assert board.loc[0,'insider_is_sell']==pytest.approx(.12)


def test_atm_selection_prefers_nearest_alpaca_expiry_then_strike():
    date=datetime(2026,6,2)
    def contract(symbol,dte,strike,right='call'):
        return dict(symbol=symbol,expiration_date=date+timedelta(days=dte),
                    strike_price=str(strike),type=right)
    contracts={
        ('A','call'):[contract('C_EXACT_EXPIRY',60,105),contract('C_EXACT_ATM_WRONG_EXPIRY',61,100)],
        ('B','put'):[contract('P_59',59,100,'put'),contract('P_61_LOWER',61,99,'put'),
                     contract('P_61_UPPER',61,101,'put')],
    }
    class Client:
        def __init__(self):self.reads=[]
        def get_option_contracts(self,symbol,**kwargs):
            self.reads.append((symbol,kwargs))
            return contracts.get((symbol,kwargs['option_type']),[])
    client=Client()
    board=pd.DataFrame({'symbol':['MISSING','A','B','NOT_ELIGIBLE'], 'rank':[1,2,3,4],
        'direction':['long','long','short','long'], 'score_date':[date]*4,'close':[100.]*4,
        'eligible':[True,True,True,False]})
    chosen,audit=runtime.select_atm_options(board,score_date='2026-06-02',target_dte=60,top_k=2,
        alpaca_client=client,option_as_of_date=date)
    assert chosen['contract_symbol'].tolist()==['C_EXACT_EXPIRY','P_61_LOWER']
    assert chosen['option_type'].tolist()==['call','put']
    assert chosen['option_data_source'].tolist()==['alpaca_live','alpaca_live']
    assert chosen['selected_by_option_ensemble'].all()
    assert audit['status'].tolist()==['missing_alpaca_contracts','selected','selected']
    assert len(client.reads)==3
    assert all(k['expiration_date_gte']=='2026-06-02' and k['expiration_date_lte']=='2026-09-15'
               for _,k in client.reads)


def test_leaderboard_uses_same_date_prices_without_refetch(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Must use saved same-date prices')
    monkeypatch.setattr(runtime,'latest_prices_from_quant_warehouse',forbidden)
    board=runtime.build_latest_equity_leaderboard(pd.DataFrame({'symbol':['A'],'date':['2026-06-02'],
        'strategy_source':['warehouse_multirate'],'long_score':[.9],'short_score':[.1]}),
        top_k=20,price_map={'A':100.})
    assert board['close'].tolist()==[100.]


def test_migrated_notebook_delegates_training_and_has_no_old_corpus_ranker_or_order_submission():
    path=Path(__file__).parents[1]/'notebooks/trading_app_v2_multirate_live.ipynb'
    notebook=json.loads(path.read_text())
    source='\n'.join(''.join(c['source']) for c in notebook['cells'])
    for cell in notebook['cells']:
        if cell['cell_type']=='code':compile(''.join(cell['source']),str(path),'exec')
    assert 'train_latest_warehouse_model(' in source
    assert 'OPTION_TENOR_DAYS = 60' in source
    assert 'select_atm_options(' in source
    assert 'SOURCE_CORPUS_PATH' not in source and 'build_multirate_mtl_corpus' not in source
    assert 'build_score_date_option_ml_ranking_table' not in source
    assert 'build_llm_ranked_option_orders' not in source
    assert 'submit_order(' not in source
    assert 'score_date = None' not in source
