## NOTE:
## You will need your own Bureau of Labor Statistics API Key in .Renviron
## prior to being able to run this code.

## ref: https://github.com/keberwein/blscrapeR

library(blscrapeR)
library(tidyverse)

# Employment Rate (Total, Men, Women) (seasonally adjusted)
emp_rate_df <- bls_api(c("LNS12300000", "LNS12300001", "LNS12300002"), 
                       startyear = 1948, endyear = 2022,
                       registrationKey = Sys.getenv("BLS_KEY")) %>%
        spread(seriesID, value) %>% dateCast()

# 16-24 (Men, Women) (seasonally adjusted)
emp_rate_16_to_24_df <- bls_api(c("LNS12324885", "LNS12324886"), 
                                startyear = 1948, endyear = 2022,
                                registrationKey = Sys.getenv("BLS_KEY")) %>%
        spread(seriesID, value) %>% dateCast()

# 25-34 (Men, Women) (seasonally adjusted)
emp_rate_25_to_34_df <- bls_api(c("LNS12300164", "LNS12300327"), 
                                startyear = 1948, endyear = 2022,
                                registrationKey = Sys.getenv("BLS_KEY")) %>%
        spread(seriesID, value) %>% dateCast()

# 35-44 (Men, Women) (seasonally adjusted)
emp_rate_35_to_44_df <- bls_api(c("LNS12300173", "LNS12300334"), 
                                startyear = 1948, endyear = 2022,
                                registrationKey = Sys.getenv("BLS_KEY")) %>%
        spread(seriesID, value) %>% dateCast()

# 45-54 (Men, Women) (seasonally adjusted)
emp_rate_45_to_54_df <- bls_api(c("LNS12300182", "LNS12300341"), 
                                startyear = 1948, endyear = 2022,
                                registrationKey = Sys.getenv("BLS_KEY")) %>%
        spread(seriesID, value) %>% dateCast()

# 55+ (Men, Women) (seasonally adjusted)
emp_rate_55_df <- bls_api(c("LNS12324231", "LNS12324232"), 
                                startyear = 1948, endyear = 2022,
                                registrationKey = Sys.getenv("BLS_KEY")) %>%
        spread(seriesID, value) %>% dateCast()

write.csv(emp_rate_df, "employmentRate.csv", row.names = FALSE)
write.csv(emp_rate_16_to_24_df, "employmentRate16to24.csv", row.names = FALSE)
write.csv(emp_rate_25_to_34_df, "employmentRate25to34.csv", row.names = FALSE)
write.csv(emp_rate_35_to_44_df, "employmentRate35to44.csv", row.names = FALSE)
write.csv(emp_rate_45_to_54_df, "employmentRate45to54.csv", row.names = FALSE)
write.csv(emp_rate_55_df, "employmentRate55+.csv", row.names = FALSE)

########################
library (readr)

raw_url = "https://raw.githubusercontent.com/automoto/big-five-data/master/big_five_scores.csv"

big_five_df <- read_csv(url(raw_url))
write.csv(big_five_df, "big_five.csv", row.names = FALSE)
