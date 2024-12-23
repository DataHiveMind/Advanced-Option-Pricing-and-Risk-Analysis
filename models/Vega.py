import numpy as np
from scipy.stats import norm
from dataclasses import dataclass

@dataclass
class Vega_models:
    stock_price : int
    strike_price : int
    time_to_marturity: int
    Risk: float
    Volatility : float
    
    def Vega_Basic(self) -> float:
        d1 = (np.log(self.stock_price/self.strike_price) + (self.Risk + 0.5 * self.Volatility ** 2) * self.time_to_marturity)/(self.Volatility * np.sqrt(self.time_to_marturity))
        vega = self.stock_price* norm.pdf(d1) * np.sqrt(self.time_to_marturity)
        return vega

if __name__ is "__main__":
    Current_stock_price = 100
    Strike_Price = 105
    Time_of_Maturity = 1 # 1 year
    Risk_Free_interest_rate = 0.05 
    Volatility = 0.2

    vega_value = Vega_models.Vega_Basic(Current_stock_price, Strike_Price, Time_of_Maturity, Risk_Free_interest_rate, Volatility)
    print(f"The Vega of the option is: {vega_value}")