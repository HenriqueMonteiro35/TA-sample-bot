import ta
import numpy as np
import pandas as pd

def get_indicators(stock, indicators):
    columns = stock.columns.to_list()
    size = len(stock["close"])
    for indicator in indicators:
        name = indicator["name"]
        for param in indicator["params"]:

            if name == "SMA":
                N = param
                if "X_SMA"+str(N) not in columns:
                    if size > N:
                        # stock["SMA"+str(N)] = stock["close"].rolling(N).mean()
                        sma = ta.trend.sma_indicator(stock["close"], n = N, fillna=False).to_numpy()
                        stock["SMA"+str(N)] = sma
                        close = stock["close"].to_numpy()
                        jumps_up = ((sma[N:] < close[N:]) & (close[N-1:-1] < sma[N-1:-1])).astype(int)
                        jumps_down = ((sma[N:] > close[N:]) & (close[N-1:-1] > sma[N-1:-1])).astype(int)
                        stock["X_SMA"+str(N)] = np.append([0]*N, jumps_up-jumps_down)
                    else:
                        stock["SMA"+str(N)] = [np.nan]*size
                        stock["X_SMA"+str(N)] = [np.nan]*size

            elif name == "EMA":
                N = param
                if "X_EMA"+str(N) not in columns:
                    if size > N:
                        # stock["EMA"+str(N)] = stock["close"].ewm(span=N).mean()
                        ema = ta.trend.ema_indicator(stock["close"], n = N, fillna = False).to_numpy()
                        stock["EMA"+str(N)] = ema
                        close = stock["close"].to_numpy()
                        jumps_up = ((ema[N:] < close[N:]) & (close[N-1:-1] < ema[N-1:-1])).astype(int)
                        jumps_down = ((ema[N:] > close[N:]) & (close[N-1:-1] > ema[N-1:-1])).astype(int)
                        stock["X_EMA"+str(N)] = np.append([0]*N, jumps_up-jumps_down)
                    else:
                        stock["EMA"+str(N)] = [np.nan]*size
                        stock["X_EMA"+str(N)] = [np.nan]*size


            elif name == "SEN":
                N = param
                if "X_SEN"+str(N) not in columns:
                    if size > N:
                        sen = ta.trend.ichimoku_base_line(stock["high"], stock["low"], n1 = N, n2 = N,
                                                            visual=False, fillna=False).to_numpy()
                        stock["SEN"+str(N)] = sen
                        close = stock["close"].to_numpy()
                        jumps_up = ((sen[N:] < close[N:]) & (close[N-1:-1] < sen[N-1:-1])).astype(int)
                        jumps_down = ((sen[N:] > close[N:]) & (close[N-1:-1] > sen[N-1:-1])).astype(int)
                        stock["X_SEN"+str(N)] = np.append([0]*N, jumps_up-jumps_down)
                    else:
                        stock["SEN"+str(N)] = [np.nan]*size
                        stock["X_SEN"+str(N)] = [np.nan]*size

            elif name == "STD":
                N = param
                if "STD"+str(N) not in columns:
                    if size > N:
                        stock["STD"+str(N)] = stock["close"].rolling(N).std()
                    else:
                        stock["STD"+str(N)] = [np.nan]*size

            elif name == "RSI":
                N_list, crosses = param
                for N in N_list:
                    if f"RSI_{N}" not in columns: # f"X_RSI_{N}_{crosses[-1]}"
                        if size > N:
                            rsi = ta.momentum.rsi(stock["close"], n = N).to_numpy()
                            stock["RSI_"+str(N)] = rsi
                            for cross in crosses:
                                jumps_up = ((rsi[N:] > cross) & (rsi[N-1:-1] <= cross)).astype(int)
                                jumps_down = ((rsi[N:] < cross) & (rsi[N-1:-1] >= cross)).astype(int)
                                stock[f"X_RSI_{N}_{cross}"] = np.append([0]*N, jumps_up-jumps_down)
                        else:
                            stock["RSI_"+str(N)] = [np.nan]*size
                            for cross in crosses:
                                stock[f"X_RSI_{N}_{cross}"] = [np.nan]*size

            elif name == "MACD":
                N1, N2, N3 = param
                if size > max(N1, N2) + N3:
                    if "MACD"+str(N1)+"_"+str(N2)+"_"+str(N3) not in columns:
                        ema1 = stock["close"].ewm(span=N1).mean().to_numpy()
                        ema2 = stock["close"].ewm(span=N2).mean().to_numpy()
                        macd = ema2 - ema1
                        ma = pd.Series(macd).rolling(N3).mean().to_numpy()
                        mask1 = (macd[N3:] > ma[N3:])
                        mask2 = (macd[N3-1:-1] < ma[N3-1:-1])
                        signal = np.zeros(len(stock)-N3)
                        signal[(mask1 & mask2)] = 1
                        signal[(~mask1 & ~mask2)] = -1
                        stock["MACD"+str(N1)+"_"+str(N2)+"_"+str(N3)] = np.append(np.zeros(N3), signal).astype(int)
                    else:
                        stock["MACD"+str(N1)+"_"+str(N2)+"_"+str(N3)] = [np.nan]*size

            elif name == "STOC":
                N_list, crosses = param
                for N in N_list:
                    if "STOC"+str(N) not in columns:
                        if size > N:
                            H = stock["high"].rolling(N).max()[N-1:]
                            L = stock["low"].rolling(N).min()[N-1:]
                            stoc = (100 * (stock["close"] - L)/(H - L)).to_numpy()
                            stock["STOC"+str(N)] = stoc
                            for cross in crosses:
                                jumps_up = ((stoc[N:] > cross) & (stoc[N-1:-1] <= cross)).astype(int)
                                jumps_down = ((stoc[N:] < cross) & (stoc[N-1:-1] >= cross)).astype(int)
                                stock[f"X_STOC{N}_{cross}"] = np.append([0]*N, jumps_up-jumps_down)
                        else:
                            stock["STOC"+str(N)] = [np.nan]*size
                            for cross in crosses:
                                stock[f"X_STOC{N}_{cross}"] = [np.nan]*size

            elif name == "SAR":
                alpha, maximum = param
                tag = str(int(1e4*alpha))+"_"+str(int(1e4*maximum))
                if "PSAR"+tag not in columns:
                    indicator = ta.trend.PSARIndicator(stock["high"], stock["low"], stock["close"],
                                                        step = alpha, max_step = maximum, fillna = False)
                    stock["PSAR"+tag] = indicator.psar().to_numpy()
                    stock["BIN_PSAR"+tag] = indicator.psar_up_indicator().to_numpy() - indicator.psar_down_indicator().to_numpy()

            elif name == "ATR":
                N = param
                if "ATR"+str(N) not in columns:
                    if size > N:
                        stock["ATR"+str(N)] = ta.volatility.average_true_range(stock["high"], stock["low"], stock["close"],
                                                                                n = N, fillna = False).to_numpy()
                    else:
                        stock["ATR"+str(N)] = [np.nan]*size

            elif name == "ADX":
                N = param
                if "ADX"+str(N) not in columns:
                    if size >= 2*N:
                        stock["ADX"+str(N)] = ta.trend.adx(stock["high"], stock["low"], stock["close"],
                                                            n = N, fillna = False).to_numpy()
                    else:
                        stock["ADX"+str(N)] = [np.nan]*size

            elif name == "CCI":
                C = 0.015
                N_list, crosses = param
                for N in N_list:
                    if "CCI"+str(N) not in columns:
                        if size > N:
                            cci = ta.trend.cci(stock["high"], stock["low"], stock["close"], n = N, c = C,
                                                fillna=False).to_numpy()
                            stock["CCI"+str(N)] = cci
                            for cross in crosses:
                                jumps_up = ((cci[N:] > cross) & (cci[N-1:-1] <= cross)).astype(int)
                                jumps_down = ((cci[N:] < cross) & (cci[N-1:-1] >= cross)).astype(int)
                                if round(cross) == 0:
                                    cross = "0"
                                stock[f"X_CCI{N}_{cross}".replace("-", "n")] = np.append([0]*N, jumps_up-jumps_down)
                        else:
                            stock["CCI"+str(N)] = [np.nan]*size
                            for cross in crosses:
                                if round(cross) == 0:
                                    cross = "0"
                                stock[f"X_CCI{N}_{cross}".replace("-", "n")] = [np.nan]*size

            elif name == "MFI":
                N_list, crosses = param
                for N in N_list:
                    if "MFI"+str(N) not in columns: # f"X_MFI{N}_{crosses[-1]}"
                        if size > N:
                            mfi = ta.volume.money_flow_index(stock["high"], stock["low"], stock["close"],
                                                                stock["voltot"], n = N, fillna=False).to_numpy()
                            stock["MFI"+str(N)] = mfi
                            for cross in crosses:
                                jumps_up = ((mfi[N:] > cross) & (mfi[N-1:-1] <= cross)).astype(int)
                                jumps_down = ((mfi[N:] < cross) & (mfi[N-1:-1] >= cross)).astype(int)
                                stock[f"X_MFI{N}_{cross}"] = np.append([0]*N, jumps_up-jumps_down)
                        else:
                            stock["MFI"+str(N)] = [np.nan]*size
                            for cross in crosses:
                                stock[f"X_MFI{N}_{cross}"] = [np.nan]*size

            elif name == "CMF":
                N_list, crosses = param
                for N in N_list:
                    if "CMF"+str(N) not in columns:
                        if size > N:
                            cmf =  ta.volume.chaikin_money_flow(stock["high"], stock["low"], stock["close"],
                                                                stock["voltot"], n = N, fillna=False).to_numpy()
                            stock["CMF"+str(N)] = cmf
                            for cross in crosses:
                                jumps_up = ((cmf[N:] > cross) & (cmf[N-1:-1] <= cross)).astype(int)
                                jumps_down = ((cmf[N:] < cross) & (cmf[N-1:-1] >= cross)).astype(int)
                                cross = round(100*cross)/100
                                if round(100*cross) == 0:
                                    cross = "00"
                                stock[f"X_CMF{N}_{cross}".replace(".", "").replace("-", "n")] = np.append([0]*N, jumps_up-jumps_down)
                        else:
                            stock["CMF"+str(N)] = [np.nan]*size
                            for cross in crosses:
                                stock[f"X_CMF{N}_{cross}".replace(".", "").replace("-", "n")] = [np.nan]*size

            elif name == "ADI":
                N = param
                if "ADI" not in columns:
                    stock["ADI"] = ta.volume.acc_dist_index(stock["high"], stock["low"], stock["close"],
                                                                stock["voltot"], fillna=False).to_numpy()

            elif name == "OBV":
                N = param
                if "OBV"+str(N) not in columns:
                    if size > N:
                        obv = ta.volume.on_balance_volume(stock["close"], stock["voltot"], fillna=False)
                        mean = ta.trend.sma_indicator(obv, n = N, fillna=False).to_numpy()
                        obv = obv.to_numpy()
                        jump_up = (obv[N:] > mean[N:]) & (obv[N-1:-1] <= mean[N-1:-1])
                        jump_down = (obv[N:] < mean[N:]) & (obv[N-1:-1] >= mean[N-1:-1])
                        stock["OBV"+str(N)] = np.append([0]*N, jump_up.astype(int) - jump_down.astype(int))
                    else:
                        stock["OBV"+str(N)] = [np.nan]*size

            elif name == "AROON":
                for N in param:
                    if "BIN_AROON"+str(N) not in columns:
                        if size > N:
                            aroon = ta.trend.AroonIndicator(stock["close"], n = N, fillna = False)
                            up, down = aroon.aroon_up().to_numpy(), aroon.aroon_down().to_numpy()
                            stock["UP_AROON"+str(N)] = up
                            stock["DOWN_AROON"+str(N)] = down
                            jump_up = (up[N:] > down[N:]) & (up[N-1:-1] <= down[N-1:-1])
                            jump_down = (up[N:] < down[N:]) & (up[N-1:-1] >= down[N-1:-1])
                            stock["BIN_AROON"+str(N)] = np.append([0]*N, jump_up.astype(int) - jump_down.astype(int))
                        else:
                            stock["UP_AROON"+str(N)] = [np.nan]*size
                            stock["DOWN_AROON"+str(N)] = [np.nan]*size
                            stock["BIN_AROON"+str(N)] = [np.nan]*size

            elif name == "AO":
                N1, N2 = param
                if f"AO{N1}_{N2}" not in columns:
                    if size > max(N1, N2):
                        ao = ta.momentum.ao(stock["high"], stock["low"], s=N1, len=N2, fillna=False).to_numpy()
                        stock[f"AO{N1}_{N2}"] = ao
                        jump_up = (ao[N2:] > 0) & (ao[N2-1:-1] <= 0)
                        jump_down = (ao[N2:] < 0) & (ao[N2-1:-1] >= 0)
                        stock[f"BIN_AO{N1}_{N2}"] = np.append([0]*N2, jump_up.astype(int) - jump_down.astype(int))
                    else:
                        stock[f"AO{N1}_{N2}"] = [np.nan]*size
                        stock[f"BIN_AO{N1}_{N2}"] = [np.nan]*size

            elif name == "UO":
                N_list, crosses = param
                for N in N_list:
                    if "UO"+str(N) not in columns: # f"X_UO{N}_{crosses[-1]}"
                        if size > 4*N + 1:
                            uo = ta.momentum.uo(stock["high"], stock["low"], stock["close"], s=N, m=2*N,
                                            len=4*N, ws=4.0, wm=2.0, wl=1.0, fillna=False).to_numpy()
                            stock["UO"+str(N)] = uo
                            for cross in crosses:
                                jumps_up = ((uo[4*N+1:] > cross) & (uo[4*N:-1] <= cross)).astype(int)
                                jumps_down = ((uo[4*N+1:] < cross) & (uo[4*N:-1] >= cross)).astype(int)
                                stock[f"X_UO{N}_{cross}"] = np.append([0]*(4*N+1), jumps_up-jumps_down)
                        else:
                            stock["UO"+str(N)] = [np.nan]*size
                            for cross in crosses:
                                stock[f"X_UO{N}_{cross}"] = [np.nan]*size

            elif name == "MEAN_CROSSOVER":
                for mean_type in ["SMA", "EMA", "SEN"]:
                    for N1 in param:
                        mean1 = stock[mean_type+str(N1)].to_numpy()
                        for N2 in param:
                            if N2 > N1:
                                tag = f"X_{mean_type+str(N1)}_{N2}"
                                if size > N2:
                                    if tag not in columns:
                                        mean2 = stock[mean_type+str(N2)].to_numpy()
                                        jumps_up = ((mean1[N2:] > mean2[N2:]) & (mean1[N2-1:-1] <= mean2[N2-1:-1])).astype(int)
                                        jumps_down = ((mean1[N2:] < mean2[N2:]) & (mean1[N2-1:-1] >= mean2[N2-1:-1])).astype(int)
                                        stock[tag] = np.append([0]*N2, jumps_up-jumps_down)
                                else:
                                    stock[tag] = [np.nan]*size

            elif name == "DEV_CROSSOVER":
                for N in param:
                    if f"X_ATR{N}_STD{N}" not in columns:
                        if size > N + 2:
                            atr = stock["ATR"+str(N)].to_numpy()
                            std = stock["STD"+str(N)].to_numpy()
                            close = stock["close"].to_numpy()

                            std_jumps = ((std[N:] > atr[N:]) & (std[N-1:-1] <= atr[N-1:-1]))
                            slope = (close[N:] > close[N-3:-3])
                            signal = (std_jumps & slope).astype(int) - (std_jumps & ~slope).astype(int)
                            stock[f"X_ATR{N}_STD{N}"] = np.append([0]*N, signal)
                        else:
                            stock[f"X_ATR{N}_STD{N}"] = [np.nan]*size

            elif name == "MARTELO":
                tipo, N1, N2 = param
                tag = f"MARTELO{tipo}_{str(N1)}_{str(N2)}"
                if tag not in columns:
                    N = N1+5 if N1 > N2 else N2+5
                    if size >= max(N+1, 2*N2):
                        std = stock["STD"+str(N1)]
                        low, high = stock["low"].copy().to_numpy(), stock["high"].copy().to_numpy()
                        opene, close = stock["open"].copy().to_numpy(), stock["close"].copy().to_numpy()

                        body = abs(close - opene)
                        u_tail = np.minimum(close, opene) - low
                        d_tail = high - np.maximum(close, opene)

                        if tipo == 1:
                            adx = stock["ADX"+str(N2)]
                            stoc = stock["STOC"+str(N2)]

                            u_mask = (u_tail[N:] > std[N:]) & (u_tail[N:] > body[N:]) & (adx[N-1:-1] > 30) & (stoc[N:] < 50) & (
                                low[N:] < pd.Series(low).rolling(5).min()[N-1:-1])
                            d_mask = (d_tail[N:] > std[N:]) & (d_tail[N:] > body[N:]) & (adx[N-1:-1] > 30) & (stoc[N:] > 50) & (
                                high[N:] > pd.Series(high).rolling(5).max()[N-1:-1])

                        elif tipo == 2:
                            adx = stock["ADX"+str(N2)]

                            u_mask = (u_tail[N:] > std[N:]) & (u_tail[N:] > body[N:]) & (adx[N-1:-1] > 30) & (
                                low[N:] < pd.Series(low).rolling(5).min()[N-1:-1])
                            d_mask = (d_tail[N:] > std[N:]) & (d_tail[N:] > body[N:]) & (adx[N-1:-1] > 30) & (
                                high[N:] > pd.Series(high).rolling(5).max()[N-1:-1])

                        elif tipo == 3:
                            stoc = stock["STOC"+str(N2)]

                            u_mask = (u_tail[N:] > std[N:]) & (u_tail[N:] > body[N:]) & (stoc[N:] < 50) & (
                                low[N:] < pd.Series(low).rolling(5).min()[N-1:-1])
                            d_mask = (d_tail[N:] > std[N:]) & (d_tail[N:] > body[N:]) & (stoc[N:] > 50) & (
                                high[N:] > pd.Series(high).rolling(5).max()[N-1:-1])

                        elif tipo == 4:
                            u_mask = (u_tail[N:] > std[N:]) & (u_tail[N:] > body[N:]) & (
                                low[N:] < pd.Series(low).rolling(5).min()[N-1:-1])
                            d_mask = (d_tail[N:] > std[N:]) & (d_tail[N:] > body[N:]) & (
                                high[N:] > pd.Series(high).rolling(5).max()[N-1:-1])

                        elif tipo == 5:
                            u_mask = (u_tail[N:] > std[N:]) & (u_tail[N:] > body[N:]) & (body[N:] > d_tail[N:]) & (
                                low[N:] < pd.Series(low).rolling(5).min()[N-1:-1])
                            d_mask = (d_tail[N:] > std[N:]) & (d_tail[N:] > body[N:]) & (body[N:] > u_tail[N:]) & (
                                high[N:] > pd.Series(high).rolling(5).max()[N-1:-1])

                        elif tipo == 6:
                            u_mask = (u_tail[N:] > std[N:]) & (std[N:] > d_tail[N:]) & (
                                low[N:] < pd.Series(low).rolling(5).min()[N-1:-1])
                            d_mask = (d_tail[N:] > std[N:]) & (std[N:] > u_tail[N:]) & (
                                high[N:] > pd.Series(high).rolling(5).max()[N-1:-1])

                        elif tipo == 7:
                            u_mask = (u_tail[N:] > std[N:]) & (std[N:] > d_tail[N:])
                            d_mask = (d_tail[N:] > std[N:]) & (std[N:] > u_tail[N:])

                        elif tipo == 8:
                            u_mask = (u_tail[N:] > body[N:]) & (low[N:] < pd.Series(low).rolling(5).min()[N-1:-1])
                            d_mask = (d_tail[N:] > body[N:]) & (high[N:] > pd.Series(high).rolling(5).max()[N-1:-1])

                        elif tipo == 9:
                            u_mask = (u_tail[N:] > 2*body[N:]) & (u_tail[N:] > 2*d_tail[N:]) & (
                                low[N:] < pd.Series(low).rolling(5).min()[N-1:-1])
                            d_mask = (d_tail[N:] > 2*body[N:]) & (d_tail[N:] > 2*u_tail[N:]) & (
                                high[N:] > pd.Series(high).rolling(5).max()[N-1:-1])

                        elif tipo == 10:
                            u_mask = (u_tail[N:] > std[N:]) & (u_tail[N:] > 2*body[N:]) & (u_tail[N:] > 2*d_tail[N:]) & (
                                low[N:] < pd.Series(low).rolling(5).min()[N-1:-1])
                            d_mask = (d_tail[N:] > std[N:]) & (d_tail[N:] > 2*body[N:]) & (d_tail[N:] > 2*u_tail[N:]) & (
                                high[N:] > pd.Series(high).rolling(5).max()[N-1:-1])

                        elif tipo == 11:
                            atr = stock["ATR"+str(N1)]
                            u_mask = (u_tail[N:] > 0.5*atr[N:]) & (u_tail[N:] > 2*body[N:]) & (u_tail[N:] > 2*d_tail[N:]) & (
                                low[N:] < pd.Series(low).rolling(5).min()[N-1:-1])
                            d_mask = (d_tail[N:] > 0.5*atr[N:]) & (d_tail[N:] > 2*body[N:]) & (d_tail[N:] > 2*u_tail[N:]) & (
                                high[N:] > pd.Series(high).rolling(5).max()[N-1:-1])

                        else:
                            raise Exception(f"MARTELO type {tipo} not found!")

                        u_martelo = (u_mask).astype(int)
                        d_martelo = (d_mask).astype(int)
                        num = len(stock) - N
                        stock[tag] = np.append([0]*N, (u_martelo-d_martelo)[:num])

                    else:
                        stock[tag] = [np.nan]*size

            elif name == "ROSS":
                tipo, width, tall = param
                num_tag = 3 if tipo == 0 else tipo
                tag = f"ROSS{tipo}_{width}_{tall}".replace(".", "")
                if tag not in columns:
                    if size > 5*width:
                        close, high, low = stock["close"], stock["high"], stock["low"]
                        atr = ta.volatility.average_true_range(high, low, close, n = width*5,
                                                                fillna = False).to_numpy()

                        tol = int(width/2)
                        dy = tall*atr[5*width:]
                        BR = close.to_numpy()[5*width:]

                        RT_L = low.rolling(tol).min().to_numpy()[4*width:-1*width]
                        RH_L = high.rolling(tol).max().to_numpy()[3*width:-2*width]
                        P3_L = low.rolling(tol).min().to_numpy()[2*width:-3*width]
                        P2_L = high.rolling(tol).max().to_numpy()[1*width:-4*width]
                        P1_L = low.rolling(tol).min().to_numpy()[:-5*width]

                        # LONG: highest close of the period at P2, RH and BR
                        mask_L = (BR > close.rolling(5*width).max().to_numpy()[5*width -1:-1])
                        mask_L = mask_L & (RH_L >= close.rolling(3*width).max().to_numpy()[3*width -1:-2*width -1])
                        mask_L = mask_L & (P2_L >= close.rolling(1*width).max().to_numpy()[1*width -1:-4*width -1])

                        RT_S = high.rolling(tol).max().to_numpy()[4*width:-1*width]
                        RH_S = low.rolling(tol).min().to_numpy()[3*width:-2*width]
                        P3_S = high.rolling(tol).max().to_numpy()[2*width:-3*width]
                        P2_S = low.rolling(tol).min().to_numpy()[1*width:-4*width]
                        P1_S = high.rolling(tol).max().to_numpy()[:-5*width]

                        # SHORT: lowest close of the period at P2, RH and BR
                        mask_S = (BR < close.rolling(5*width).min().to_numpy()[5*width -1:-1])
                        mask_S = mask_S & (RH_S <= close.rolling(3*width).min().to_numpy()[3*width -1:-2*width -1])
                        mask_S = mask_S & (P2_S <= close.rolling(1*width).min().to_numpy()[1*width -1:-4*width -1])

                        if tipo in [0, 3]:
                            tag = f"ROSS3_{width}_{tall}".replace(".", "")
                            # LONG
                            # BR above RT
                            mask_L = mask_L & (BR > RT_L + dy)
                            # RH above P2 and much above P3
                            mask_L = mask_L & (RH_L > P3_L + 2*dy)

                            # SHORT
                            # BR below RT
                            mask_S = mask_S & (BR < RT_S - dy)
                            # RH below P2 and much below P3
                            mask_S = mask_S & (RH_S < P3_S - 2*dy)

                            signal = mask_L.astype(int) - mask_S.astype(int)
                            stock[tag] = np.append([0]*(5*width), signal)

                        if tipo in [0, 2]:
                            tag = f"ROSS2_{width}_{tall}".replace(".", "")
                            # LONG
                            # BR above RT
                            mask_L = mask_L & (BR > RT_L + dy)
                            # RH above P2 and much above P3
                            mask_L = mask_L & (RH_L > P3_L + 2*dy)
                            mask_L = mask_L & (RH_L > P2_L + dy)

                            # SHORT
                            # BR below RT
                            mask_S = mask_S & (BR < RT_S - dy)
                            # RH below P2 and much below P3
                            mask_S = mask_S & (RH_S < P3_S - 2*dy)
                            mask_S = mask_S & (RH_S < P2_S - dy)

                            signal = mask_L.astype(int) - mask_S.astype(int)
                            stock[tag] = np.append([0]*(5*width), signal)

                        if tipo in [0, 1]:
                            tag = f"ROSS1_{width}_{tall}".replace(".", "")
                            # LONG
                            # BR above RT
                            mask_L = mask_L & (BR > RT_L + dy)
                            # RH above P2 and much above P3
                            mask_L = mask_L & (RH_L > P3_L + 2*dy)
                            mask_L = mask_L & (RH_L > P2_L + dy)
                            # P3 above P1
                            mask_L = mask_L & (P3_L > P1_L + dy)
                            # P2 much above P1
                            mask_L = mask_L & (P2_L > P1_L + 2*dy)

                            # SHORT
                            # BR below RT
                            mask_S = mask_S & (BR < RT_S - dy)
                            # RH below P2 and much below P3
                            mask_S = mask_S & (RH_S < P3_S - 2*dy)
                            mask_S = mask_S & (RH_S < P2_S - dy)
                            # P3 below P1
                            mask_S = mask_S & (P3_S < P1_S - dy)
                            # P2 much below P1
                            mask_S = mask_S & (P2_S < P1_S - 2*dy)

                            signal = mask_L.astype(int) - mask_S.astype(int)
                            stock[tag] = np.append([0]*(5*width), signal)
                    else:
                        stock[f"ROSS1_{width}_{tall}".replace(".", "")] = [np.nan]*size
                        stock[f"ROSS2_{width}_{tall}".replace(".", "")] = [np.nan]*size
                        stock[f"ROSS3_{width}_{tall}".replace(".", "")] = [np.nan]*size


            elif name == "3BP":
                tipo, mult1, mult2 = param
                tag = f"3BP{tipo}_{mult1}_{mult2}".replace(".", "")
                if tag not in columns:
                    if size > 13:
                        N = 13
                        atr, change = stock["ATR10"].to_numpy(), stock["close"].to_numpy() - stock["open"].to_numpy()
                        mask_L = (change[N:] > mult1*atr[N:])
                        mask_S = (-change[N:] > mult1*atr[N:])

                        if tipo == 1:
                            # LONG CASE 1
                            mask_L1 = (change[N-1:-1] < 0) & (abs(change[N-1:-1]) < mult2*atr[N-1:-1])
                            mask_L1 = mask_L1 & (change[N-2:-2] > mult1*atr[N-2:-2])
                            # LONG CASE 2
                            mask_L2 = (change[N-1:-1] > 0) & (change[N-1:-1] < mult2*atr[N-1:-1])
                            mask_L2 = mask_L2 & (change[N-2:-2] < 0) & (abs(change[N-2:-2]) < mult2*atr[N-2:-2])
                            mask_L2 = mask_L2 & (change[N-3:-3] > mult1*atr[N-3:-3])

                            # SHORT CASE 1
                            mask_S1 = (change[N-1:-1] > 0) & (change[N-1:-1] < mult2*atr[N-1:-1])
                            mask_S1 = mask_S1 & (change[N-2:-2] < 0) & (abs(change[N-2:-2]) > mult1*atr[N-2:-2])
                            # SHORT CASE 2
                            mask_S2 = (change[N-1:-1] < 0) & (abs(change[N-1:-1]) < mult2*atr[N-1:-1])
                            mask_S2 = mask_S2 & (change[N-2:-2] > 0) & (change[N-2:-2] < mult2*atr[N-2:-2])
                            mask_S2 = mask_S2 & (change[N-3:-3] < 0) & (abs(change[N-3:-3]) > mult1*atr[N-3:-3])

                        elif tipo == 2:
                            ATR = atr[N-3:-3]
                            # LONG CASE 1
                            mask_L1 = (change[N-1:-1] < 0) & (abs(change[N-1:-1]) < mult2*ATR)
                            mask_L1 = mask_L1 & (change[N-2:-2] > mult1*ATR)
                            mask_L1 = mask_L1 & (change[N-2:-2] > change[N:]) # new
                            # LONG CASE 2
                            mask_L2 = (change[N-1:-1] > 0) & (change[N-1:-1] < mult2*ATR)
                            mask_L2 = mask_L2 & (change[N-2:-2] < 0) & (abs(change[N-2:-2]) < mult2*ATR)
                            mask_L2 = mask_L2 & (change[N-3:-3] > mult1*ATR)
                            mask_L2 = mask_L2 & (change[N-3:-3] > change[N:]) # new

                            # SHORT CASE 1
                            mask_S1 = (change[N-1:-1] > 0) & (change[N-1:-1] < mult2*ATR)
                            mask_S1 = mask_S1 & (change[N-2:-2] < 0) & (abs(change[N-2:-2]) > mult1*ATR)
                            mask_S1 = mask_S1 & (abs(change[N-2:-2]) > abs(change[N:])) # new
                            # SHORT CASE 2
                            mask_S2 = (change[N-1:-1] < 0) & (abs(change[N-1:-1]) < mult2*ATR)
                            mask_S2 = mask_S2 & (change[N-2:-2] > 0) & (change[N-2:-2] < mult2*ATR)
                            mask_S2 = mask_S2 & (change[N-3:-3] < 0) & (abs(change[N-3:-3]) > mult1*ATR)
                            mask_S2 = mask_S2 & (abs(change[N-3:-3]) > abs(change[N:])) # new

                        mask_L = mask_L & (mask_L1 | mask_L2)
                        mask_S = mask_S & (mask_S1 | mask_S2)
                        signal = mask_L.astype(int) - mask_S.astype(int)
                        stock[tag] = np.append([0]*N, signal)
                    else:
                        stock[tag] = [np.nan]*size

            else:
                raise Exception(f"Indicator {name} not found!")
    return stock