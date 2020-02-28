import fbprophet
import pytrends
from fbprophet import Prophet
import numpy as np
from pytrends.request import TrendReq
from pytrends.exceptions import ResponseError
import matplotlib.pyplot as plt
pytrends = TrendReq(hl='en-US', tz=360)

class Trend_Forecast:

  def __init__(self, keyword, group, loc, year_start, month_start, day_start, year_end, month_end, day_end, num_future_days):
    '''
    keyword: keyword to be searched on Google
    group: the google platform on which the google search take place; options: images, youtube, news
    loc: geographical location in which the google searches take place
    '''

    self.keyword = keyword
    self.group = group 
    self.loc = loc
    self.year_start = year_start
    self.month_start = month_start 
    self.day_start = day_start
    self.year_end = year_end
    self.month_end = month_end 
    self.day_end = day_end
    self.num_future_days = num_future_days

  def fetch_from_google(self):

    pytrends = TrendReq(hl='en-US', tz=360)
    #pytrends.build_payload([self.keyword], cat=0, timeframe='today 5-y', geo='', gprop=self.group)

    attempts, collected = 0, False
    while not collected:
        try:
            pytrends.build_payload([self.keyword], cat=0, timeframe='today 5-y', geo='', gprop=self.group)
        except ResponseError as err:
            print(err)
            print(f'Trying again in {40 + 5 * attempts} seconds.')
            sleep(40 + 5 * attempts)
            attempts += 1
            if attempts > 3:
                print('Failed after 3 attemps, abort fetching.')
                break
        else:
            collected = True

    data = pytrends.get_historical_interest([self.keyword], 
                                            year_start=self.year_start, month_start=self.month_start, day_start=self.day_start, 
                                            year_end=self.year_end, month_end=self.month_end, day_end=self.day_end, 
                                            cat=0, geo='', gprop=self.group, sleep=0)
    data.drop(columns=['isPartial'], inplace=True)
    daily_data = data.groupby(data.index.date).sum()
    daily_data = daily_data[1:-1]
    daily_data[self.keyword] = daily_data[self.keyword]/daily_data[self.keyword].max()

    return daily_data

  def prophet(self):

    daily_data = self.fetch_from_google()
    prophet_data = daily_data.copy()
    prophet_data["ds"] = prophet_data.index
    prophet_data["y"] = prophet_data[self.keyword]
    prophet_model = Prophet()
    prophet_model.fit(prophet_data)
    future_data = prophet_model.make_future_dataframe(periods = self.num_future_days)
    forecast = prophet_model.predict(future_data)
    
    yhat = forecast["yhat"]
    ylow = forecast["yhat_lower"]
    yhigh = forecast["yhat_upper"]
    
    summary = {"input time" : daily_data.index.values, 
               "input value": daily_data[self.keyword].values, 
               "output time": future_data.ds.values,
               "output value": yhat, 
               "output lower": ylow, 
               "output higher": yhigh}

    return summary

  def gp_forecast(self):

    k2 = 2.4**2 * kernels.ExpSquaredKernel(90**2) * kernels.ExpSine2Kernel(2.0 / 1.3**2, 1.0)
    kernel = k2

    daily_data = self.fetch_from_google()
    daily_data = daily_data.asfreq('D', method='pad')
    future_daily_data = daily_data.tshift(self.num_future_days)
    
    present_time = 2000 + (np.array(daily_data.index.to_julian_date()) - 2451545.0) / 365.25
    future_time = 2000 + (np.array(future_daily_data.index.to_julian_date()) - 2451545.0) / 365.25
    y = np.array(daily_data[self.keyword])

    gp = george.GP(kernel, mean=np.mean(y), fit_mean=True,
               white_noise=np.log(0.19**2), fit_white_noise=True)
    gp.compute(present_time)
    
    print("current log-likelihood value", gp.log_likelihood(y))
    print("current derivative of the log-likelihood ", gp.grad_log_likelihood(y))


    def nll(p):
      gp.set_parameter_vector(p)
      ll = gp.log_likelihood(y, quiet=True)
      return -ll if np.isfinite(ll) else 1e25

    # And the gradient of the objective function.
    def grad_nll(p):
      gp.set_parameter_vector(p)
      return -gp.grad_log_likelihood(y, quiet=True)

    # Run the optimization routine.
    p0 = gp.get_parameter_vector()
    results = op.minimize(nll, p0, jac=grad_nll, method="L-BFGS-B")

    # Update the kernel and print the final log-likelihood.
    gp.set_parameter_vector(results.x)

    ypred, ycov = gp.predict(y, future_time)
    std = np.sqrt(np.diag(ycov))

    summary = {"input time" : daily_data.index.values, 
               "input value": daily_data[self.keyword].values, 
               "output time": future_daily_data.index.values,
               "output value": ypred, 
               "output lower": ypred - std, 
               "output higher": ypred + std}

    return summary


  def plot(self, model_choice):

    if model_choice == "prophet":

      summary = self.prophet()

    elif model_choice == "GP":

      summary = self.gp_forecast()

    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (30, 30))
    ax.scatter(summary["input time"], summary["input value"], s = 50, color = "k", alpha = 0.6)
    ax.fill_between(summary["output time"], summary["output lower"], summary["output higher"], alpha = 0.6)
    ax.plot(summary["output time"], summary["output value"], linewidth = 10, color = "C3")
    ax.set_xlabel("Date" , fontsize = 100)
    ax.set_ylabel("Google trend forecast "+self.keyword, fontsize = 100)
    

    return fig

