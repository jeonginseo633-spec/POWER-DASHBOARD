"""PDF 기반 전력 예측. 실행 예시는 사용방법.md를 참고하세요."""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import ExtraTreesRegressor
BASE = Path(__file__).resolve().parent

COLS=['생산량','기온','풍속','습도','강수량','전기요금(계절)','공장인원','인건비']
def features(df):
    dt=df.datetime; h=dt.dt.hour.astype(float); dow=dt.dt.dayofweek.astype(float); doy=dt.dt.dayofyear.astype(float)
    q=df['생산량'].fillna(0);t=df['기온'].interpolate(limit_direction='both');wind=df['풍속'].interpolate(limit_direction='both');hum=df['습도'].interpolate(limit_direction='both');rain=df['강수량'].fillna(0);workers=df['공장인원'].fillna(0);work=(workers>0).astype(float);heat=np.maximum(18-t,0);cool=np.maximum(t-22,0)
    x=pd.DataFrame(dict(production=q,temp=t,heat_deg18=heat,cool_deg22=cool,wind=wind,humidity=hum,rain=rain,tariff=df['전기요금(계절)'].fillna(0),workers=workers,workers_missing=df['공장인원'].isna().astype(float),labor_cost=df['인건비'].fillna(0)))
    for k in [1,2,3]:
        suffix='' if k==1 else str(k)
        x['sin_hour'+suffix]=np.sin(k*2*np.pi*h/24);x['cos_hour'+suffix]=np.cos(k*2*np.pi*h/24)
    x['sin_week']=np.sin(2*np.pi*(dow*24+h)/168);x['cos_week']=np.cos(2*np.pi*(dow*24+h)/168)
    x['sin_year']=np.sin(2*np.pi*doy/365.25);x['cos_year']=np.cos(2*np.pi*doy/365.25)
    x['production_work']=q*work;x['cool_work']=cool*work;x['hour']=h;x['dow']=dow;x['day_of_year']=doy;x['month']=dt.dt.month.astype(float)
    day=dt.dt.floor('D');week=dt.dt.to_period('W-SUN')
    for name,s,key,fn in [('prod_day_sum',q,day,'sum'),('prod_day_max',q,day,'max'),('workers_day_sum',workers,day,'sum'),('workers_day_max',workers,day,'max'),('active_hours_day',work,day,'sum'),('prod_week_sum',q,week,'sum'),('workers_week_sum',workers,week,'sum'),('active_hours_week',work,week,'sum'),('rows_in_week',q,week,'size')]:x[name]=s.groupby(key).transform(fn)
    x['working']=work;x['zero_production']=(q==0).astype(float);x['zero_operations']=((q==0)&(workers==0)).astype(float)
    x['zero_operation_day']=((x.prod_day_sum==0)&(x.workers_day_sum==0)).astype(float)
    x['shutdown_week']=((x.rows_in_week==168)&(x.prod_week_sum==0)&(x.workers_week_sum==0)).astype(float)
    x['is_weekend']=(dow>=5).astype(float);x['slot']=dow*24+h;x['log1p_production']=np.log1p(np.maximum(q,0));x['sqrt_production']=np.sqrt(np.maximum(q,0));x['temp_work']=t*work
    return x.astype(float)
def metric(y,p):
    y=np.asarray(y);p=np.asarray(p);e=p-y;nz=abs(y)>1e-9
    return dict(n=len(y),mae=float(abs(e).mean()),rmse=float(np.sqrt((e*e).mean())),r2=float(1-(e*e).sum()/((y-y.mean())**2).sum()),mape=float(np.mean(abs(e[nz]/y[nz]))*100))

class PowerPredictor:
    """웹 계산기와 같은 학습 설정 및 과거 패턴 가정을 사용합니다."""
    def fit(self, csv_path):
        d = pd.read_csv(csv_path)
        if not d.groupby('날짜', sort=False).size().eq(24).all():
            raise ValueError('시간 복원을 위해 날짜별 24행이 필요합니다.')
        h = d.groupby('날짜', sort=False).cumcount()
        if not ((d['시간'] == h) | ~d['시간'].between(0,23)).all():
            raise ValueError('시간과 행 순서가 일치하지 않습니다.')
        d['datetime'] = pd.to_datetime(d['날짜'].astype(str)) + pd.to_timedelta(h, unit='h')
        d = d.sort_values('datetime').reset_index(drop=True)
        train, test = d[d.datetime < '2021-07-01'].copy(), d[d.datetime >= '2021-07-01'].copy()
        self.model = ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2,
                                        max_features=.75, random_state=42, n_jobs=-1)
        self.model.fit(features(train), train['평균'])
        quiet = train[(train['생산량'].fillna(0)==0) & (train['공장인원'].fillna(0)==0)]
        self.standby = quiet.groupby(quiet.datetime.dt.hour)['평균'].median().reindex(range(24)).fillna(quiet['평균'].median()).to_numpy()
        clean = train[COLS].fillna(train[COLS].median()).fillna(0)
        self.profiles = {}
        for dow in range(7):
            for hour in range(24):
                self.profiles[dow*24+hour] = clean.loc[(train.datetime.dt.dayofweek==dow)&(train.datetime.dt.hour==hour)].median().to_numpy()
        x = features(test)
        pred = self.model.predict(x)
        mask = x.shutdown_week.eq(1).to_numpy()
        pred[mask] = self.standby[test.datetime.dt.hour.to_numpy()[mask]]
        self.metrics = metric(test['평균'], pred)
        test = test[['datetime','평균']].copy()
        test['prediction'] = pred
        return test

    def predict_day(self, date, hour=11, production=None, temperature=None,
                    workers=None, weekday=None, shutdown=False, rate=None):
        """선택 시간 입력을 반영해 24시간을 재계산합니다. weekday: 월=1~일=7."""
        target = pd.Timestamp(date).normalize()
        if not isinstance(hour, int) or not 0 <= hour <= 23:
            raise ValueError('시간은 0~23 정수입니다.')
        if weekday is not None and (weekday not in range(1,8)):
            raise ValueError('요일은 월=1~일=7입니다.')
        for name, value in [('production', production),('temperature', temperature),('workers',workers),('rate',rate)]:
            if value is not None and (not np.isfinite(value) or (name != 'temperature' and value < 0)):
                raise ValueError('입력값 범위를 확인하세요.')
        start = target - pd.Timedelta(days=target.dayofweek)
        dates = pd.date_range(start, periods=168, freq='h')
        shift = 0 if weekday is None else weekday-1-target.dayofweek
        dow = (dates.dayofweek + shift) % 7
        rows = [self.profiles[int(w*24+h)] for w,h in zip(dow, dates.hour)]
        d = pd.DataFrame(rows, columns=COLS)
        d['datetime'] = dates
        selected = (dates.normalize()==target) & (dates.hour==hour)
        for col,value in [('생산량',production),('기온',temperature),('공장인원',workers)]:
            if value is not None: d.loc[selected,col] = value
        if shutdown: d[['생산량','공장인원']] = 0
        x = features(d)
        # 요일 시나리오만 변경하고 실제 날짜의 월·연중 일수는 유지합니다.
        slot = dow*24+dates.hour
        x['dow']=dow; x['slot']=slot; x['is_weekend']=(dow>=5).astype(float)
        x['sin_week']=np.sin(2*np.pi*slot/168); x['cos_week']=np.cos(2*np.pi*slot/168)
        pred = self.model.predict(x)
        mask=x.shutdown_week.eq(1).to_numpy()
        pred[mask]=self.standby[dates.hour[mask]]
        out=d.loc[dates.normalize()==target,['datetime','생산량','기온','공장인원']].copy()
        out['predicted_kW']=pred[dates.normalize()==target]
        out['energy_kWh']=out.predicted_kW  # 각 행은 1시간의 평균 전력
        if rate is not None: out['usage_cost_won']=out.energy_kWh*rate
        return out.reset_index(drop=True)

def main():
    parser=argparse.ArgumentParser(description='전력 예측: 학습 또는 날짜별 계산')
    parser.add_argument('--train',action='store_true')
    parser.add_argument('--csv',type=Path,default=BASE/'okm_augumented_2021.csv')
    parser.add_argument('--model',type=Path,default=BASE/'power_model.joblib')
    parser.add_argument('--date',default='2026-12-28')
    parser.add_argument('--hour',type=int,default=11)
    parser.add_argument('--production',type=float)
    parser.add_argument('--temperature',type=float)
    parser.add_argument('--workers',type=float)
    parser.add_argument('--weekday',type=int)
    parser.add_argument('--shutdown',action='store_true')
    parser.add_argument('--rate',type=float)
    args=parser.parse_args()
    if args.train or not args.model.exists():
        predictor=PowerPredictor()
        validation=predictor.fit(args.csv)
        # 사전만 저장하여 스크립트 실행/모듈 import 양쪽에서 로드할 수 있게 합니다.
        joblib.dump(predictor.__dict__,args.model,compress=3)
        validation.to_csv(BASE/'validation.csv',index=False,encoding='utf-8-sig')
        (BASE/'metrics.json').write_text(json.dumps(predictor.metrics,indent=2),encoding='utf8')
    else:
        predictor=PowerPredictor()
        predictor.__dict__.update(joblib.load(args.model))
    out=predictor.predict_day(args.date,args.hour,args.production,args.temperature,args.workers,args.weekday,args.shutdown,args.rate)
    out.to_csv(BASE/'prediction.csv',index=False,encoding='utf-8-sig')
    print(out.to_string(index=False))
    print('Daily energy (kWh):',round(out.energy_kWh.sum(),2))
    print('Conditional estimate. Oct-Dec accuracy is unverified; missing inputs use Jan-Jun profiles.')

if __name__=='__main__':
    main()
