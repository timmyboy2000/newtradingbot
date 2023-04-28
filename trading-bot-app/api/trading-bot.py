import alpaca_trade_api as tradeapi
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from finrl.config import config
from finrl.marketdata.yahoodownloader import YahooDownloader
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.preprocessing.data import data_split
from finrl.env.env_stocktrading import StockTradingEnv
from finrl.model.models import DRLAgent

# Replace with your Alpaca API key and secret key
API_KEY = 'PKB993ZIESM6J39NWM1Z'
SECRET_KEY = '7iAnLZfbkSVB67Jsx2GO6F0T4hYzdVJx5o0Kkjtp'

api = tradeapi.REST(API_KEY, SECRET_KEY, base_url='https://paper-api.alpaca.markets', api_version='v2')

def get_historical_data(stock, start_date, end_date, timeframe='1D'):
    return api.get_barset(stock, timeframe, start=start_date, end=end_date).df[stock]

symbol = 'AAPL'
start_date = '2021-01-01'
end_date = '2021-12-31'

historical_data = get_historical_data(symbol, start_date, end_date)
historical_data.reset_index(inplace=True)
historical_data.rename(columns={"index": "date"}, inplace=True)

# LSTM stock price prediction
def preprocess_data(data, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

lookback = 30
data = historical_data[['close']].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

X, y = preprocess_data(scaled_data, lookback)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.reshape(-1, lookback, 1)
X_test = X_test.reshape(-1, lookback, 1)

model = Sequential()
model.add(LSTM(units=128, activation='relu', input_shape=(lookback, 1), return_sequences=True))
model.add(LSTM(units=64, activation='relu', return_sequences=False))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

historical_data['predicted_close'] = np.nan
historical_data.loc[lookback:, 'predicted_close'] = scaler.inverse_transform(model.predict(scaled_data[:-lookback].reshape(-1, lookback, 1)))

# Reinforcement learning trading bot
fe = FeatureEngineer(
    historical_data,
    use_technical_indicator=True,
    tech_indicator_list=config.TECHNICAL_INDICATORS_LIST + ['predicted_close'],
    use_turbulence=True,
    user_defined_feature=False,
)

processed_data = fe.preprocess_data()
train_data, trade_data = data_split(processed_data, start_date, end_date)

stock_dimension = len(train_data.tic.unique())
state_space = 1 + 2*stock_dimension + len(config.TECHNICAL_INDICATORS_LIST + ['predicted_close'])*stock
_dimension
env_kwargs = {
"hmax": 100,
"initial_amount": 1000000,
"buy_cost_pct": 0.001,
"sell_cost_pct": 0.001,
"state_space": state_space,
"stock_dim": stock_dimension,
"tech_indicator_list": config.TECHNICAL_INDICATORS_LIST + ['predicted_close'],
"action_space": stock_dimension,
"reward_scaling": 1e-4,
}

e_train_gym = StockTradingEnv(df=train_data, **env_kwargs)
e_trade_gym = StockTradingEnv(df=trade_data, **env_kwargs)

agent = DRLAgent(env=e_train_gym)

ppo_params = {
'n_steps': 2048,
'ent_coef': 0.01,
'learning_rate': 0.00025,
'batch_size': 128,
'gae_lambda': 0.95,
'gamma': 0.99,
'n_epochs': 10,
'clip_range': 0.2,
'clip_range_vf': None,
'vf_coef': 0.5,
'max_grad_norm': 0.5,
'seed': None,
'verbose': 0,
'n_cpu_tf_sess': None,
'tensorboard_log': None,
'_init_setup_model': True,
}

model_ppo = agent.get_model("ppo", model_kwargs=ppo_params)
trained_ppo = agent.train_model(model=model_ppo, tb_log_name='ppo', total_timesteps=200000)
Trading
trade_df = e_trade_gym.df
e_trade_gym.reset()
obs = e_trade_gym.reset()

for i in range(len(trade_df.index.unique())):
action, _states = trained_ppo.predict(obs)
obs, rewards, done, info = e_trade_gym.step(action)
if done:
break

print("Final portfolio value: {}".format(e_trade_gym.portfolio_value))