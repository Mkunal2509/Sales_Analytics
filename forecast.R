suppressPackageStartupMessages({
  library(forecast)
})

args <- commandArgs(trailingOnly = TRUE)
input_file <- ifelse(length(args) > 0, args[1], "temp_daily_sales.csv")
horizon <- ifelse(length(args) > 1, as.numeric(args[2]), 30)

data <- read.csv(input_file)
sales_ts <- ts(data$Revenue, frequency = 7)

fit <- auto.arima(sales_ts)
fc <- forecast(fit, h = horizon)

result <- data.frame(
  Day = seq(1, horizon),
  Forecast = pmax(0, as.numeric(fc$mean)),
  Lo80 = pmax(0, as.numeric(fc$lower[, 1])),
  Hi80 = pmax(0, as.numeric(fc$upper[, 1])),
  Lo95 = pmax(0, as.numeric(fc$lower[, 2])),
  Hi95 = pmax(0, as.numeric(fc$upper[, 2]))
)

write.csv(result, row.names = FALSE)