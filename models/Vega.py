import numpy as np
from scipy.stats import norm

def Vega(S, K, T, R, Sigma) -> float:
    d1 = (np.log(S/K) + (R + 0.5 * Sigma ** 2) * T)/(Sigma * np.sqrt(T))
    vega = S* norm.pdf(d1) * np.sqrt(T)
    return vega

if __name__ is "__main__":
    Current_stock_price = 100
    Strike_Price = 105
    Time_of_Maturity = 1 # 1 year
    Risk_Free_interest_rate = 0.05 
    Volatility = 0.2

    vega_value = Vega(Current_stock_price, Strike_Price, Time_of_Maturity, Risk_Free_interest_rate, Volatility)
    print(f"The Vega of the option is: {vega_value}")