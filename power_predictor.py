import streamlit as st
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import ExtraTreesRegressor

# 페이지 기본 설정
st.set_page_config(page_title="전력 예측 대시보드", page_icon="⚡", layout="wide")

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
    def fit(self, csv_path):
        d = pd.read_csv(csv_path)
        h = d.groupby('날짜', sort=False).cumcount()
        d['datetime'] = pd.to_datetime(d['날짜'].astype(str)) + pd.to_timedelta(h, unit='h')
        d = d.sort_values('datetime').reset_index(drop=True)
        train, test = d[d.datetime < '2021-07-01'].copy(), d[d.datetime >= '2021-07-01'].copy()
        self.model = ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, max_features=.75, random_state=42, n_jobs=-1)
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
        return test

    def predict_day(self, date, hour=11, production=None, temperature=None, workers=None, weekday=None, shutdown=False, rate=None):
        target = pd.Timestamp(date).normalize()
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
        slot = dow*24+dates.hour
        x['dow']=dow; x['slot']=slot; x['is_weekend']=(dow>=5).astype(float)
        x['sin_week']=np.sin(2*np.pi*slot/168); x['cos_week']=np.cos(2*np.pi*slot/168)
        pred = self.model.predict(x)
        mask=x.shutdown_week.eq(1).to_numpy()
        pred[mask]=self.standby[dates.hour[mask]]
        out=d.loc[dates.normalize()==target,['datetime','생산량','기온','공장인원']].copy()
        out['predicted_kW']=pred[dates.normalize()==target]
        out['energy_kWh']=out.predicted_kW 
        if rate is not None: out['usage_cost_won']=out.energy_kWh*rate
        return out.reset_index(drop=True)

# 모델 불러오기 및 캐싱 (로딩 속도 최적화)
@st.cache_resource
def load_model():
    predictor = PowerPredictor()
    csv_path = BASE / 'okm_augumented_2021.csv'
    model_path = BASE / 'power_model.joblib'
    
    if model_path.exists():
        predictor.__dict__.update(joblib.load(model_path))
    else:
        with st.spinner('최초 모델 학습 중입니다. 잠시만 기다려주세요... (약 1~2분 소요)'):
            predictor.fit(csv_path)
            joblib.dump(predictor.__dict__, model_path, compress=3)
    return predictor

# 화면 구성
st.title("⚡ AI 기반 전력 예측 대시보드")
st.markdown("과거 패턴을 분석하여 설정된 조건에 따른 24시간 전력 사용량을 예측합니다.")

try:
    predictor = load_model()
    
    # 사이드바 입력 설정
    st.sidebar.header("⚙️ 예측 조건 설정")
    in_date = st.sidebar.date_input("날짜 선택", pd.to_datetime('2026-12-28'))
    in_hour = st.sidebar.slider("시간 선택", 0, 23, 11)
    in_prod = st.sidebar.number_input("예상 생산량", value=100.0)
    in_temp = st.sidebar.number_input("예상 기온 (℃)", value=20.0)
    in_work = st.sidebar.number_input("예상 공장인원", value=50.0)
    
    if st.sidebar.button("결과 예측하기", type="primary"):
        with st.spinner("데이터 예측 중..."):
            result = predictor.predict_day(
                date=str(in_date), 
                hour=in_hour, 
                production=in_prod, 
                temperature=in_temp, 
                workers=in_work
            )
            
            total_energy = round(result.energy_kWh.sum(), 2)
            
            st.success("예측이 완료되었습니다!")
            st.metric(label="일일 총 예상 전력량", value=f"{total_energy:,} kWh")
            
            st.subheader("📈 시간대별 전력 예측 그래프")
            chart_data = result.set_index('datetime')[['predicted_kW']]
            st.line_chart(chart_data)
            
            st.subheader("📋 상세 데이터")
            st.dataframe(result, use_container_width=True)

except Exception as e:
    st.error(f"데이터를 불러오는 중 문제가 발생했습니다. (에러: {e}) csv 파일이 깃허브에 있는지 확인해주세요.")
