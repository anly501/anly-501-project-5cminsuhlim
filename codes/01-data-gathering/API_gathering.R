## NOTE:
## You will need your own Bureau of Labor Statistics API Key in .Renviron
## prior to being able to run this code.

## ref: https://github.com/keberwein/blscrapeR

library(blscrapeR)
library(tidyverse)

# Employment Rate (Total, Men, Women) (seasonally adjusted)
emp_rate_df1 <- bls_api(c("LNS12300000", "LNS12300001", "LNS12300002"), 
                       startyear = 1948, endyear = 1967,
                       registrationKey = Sys.getenv("BLS_KEY")) %>%
        spread(seriesID, value) %>% dateCast()
emp_rate_df2 <- bls_api(c("LNS12300000", "LNS12300001", "LNS12300002"), 
                        startyear = 1968, endyear = 1987,
                        registrationKey = Sys.getenv("BLS_KEY")) %>%
        spread(seriesID, value) %>% dateCast()
emp_rate_df3 <- bls_api(c("LNS12300000", "LNS12300001", "LNS12300002"), 
                        startyear = 1988, endyear = 2007,
                        registrationKey = Sys.getenv("BLS_KEY")) %>%
        spread(seriesID, value) %>% dateCast()
emp_rate_df4 <- bls_api(c("LNS12300000", "LNS12300001", "LNS12300002"), 
                        startyear = 2008, endyear = 2022,
                        registrationKey = Sys.getenv("BLS_KEY")) %>%
        spread(seriesID, value) %>% dateCast() %>% select(-latest)
emp_rate_df <- rbind(emp_rate_df1, emp_rate_df2, emp_rate_df3, emp_rate_df4)


# 16-24 (Men, Women) (seasonally adjusted)
emp_rate_16_to_24_df1 <- bls_api(c("LNS12324885", "LNS12324886"), 
                                startyear = 1948, endyear = 1967,
                                registrationKey = Sys.getenv("BLS_KEY")) %>%
        spread(seriesID, value) %>% dateCast()
emp_rate_16_to_24_df2 <- bls_api(c("LNS12324885", "LNS12324886"), 
                                 startyear = 1968, endyear = 1987,
                                 registrationKey = Sys.getenv("BLS_KEY")) %>%
        spread(seriesID, value) %>% dateCast()
emp_rate_16_to_24_df3 <- bls_api(c("LNS12324885", "LNS12324886"), 
                                 startyear = 1988, endyear = 2007,
                                 registrationKey = Sys.getenv("BLS_KEY")) %>%
        spread(seriesID, value) %>% dateCast()
emp_rate_16_to_24_df4 <- bls_api(c("LNS12324885", "LNS12324886"), 
                                 startyear = 2008, endyear = 2022,
                                 registrationKey = Sys.getenv("BLS_KEY")) %>%
        spread(seriesID, value) %>% dateCast() %>% select(-latest)
emp_rate_16_to_24_df <- rbind(emp_rate_16_to_24_df1, 
                              emp_rate_16_to_24_df2, 
                              emp_rate_16_to_24_df3, 
                              emp_rate_16_to_24_df4)


# 25-34 (Men, Women) (seasonally adjusted)
emp_rate_25_to_34_df1 <- bls_api(c("LNS12300164", "LNS12300327"), 
                                 startyear = 1948, endyear = 1967,
                                 registrationKey = Sys.getenv("BLS_KEY")) %>%
        spread(seriesID, value) %>% dateCast()
emp_rate_25_to_34_df2 <- bls_api(c("LNS12300164", "LNS12300327"), 
                                 startyear = 1968, endyear = 1987,
                                 registrationKey = Sys.getenv("BLS_KEY")) %>%
        spread(seriesID, value) %>% dateCast()
emp_rate_25_to_34_df3 <- bls_api(c("LNS12300164", "LNS12300327"), 
                                 startyear = 1988, endyear = 2007,
                                registrationKey = Sys.getenv("BLS_KEY")) %>%
        spread(seriesID, value) %>% dateCast()
emp_rate_25_to_34_df4 <- bls_api(c("LNS12300164", "LNS12300327"), 
                                 startyear = 2008, endyear = 2022,
                                 registrationKey = Sys.getenv("BLS_KEY")) %>%
        spread(seriesID, value) %>% dateCast() %>% select(-latest)
emp_rate_25_to_34_df <- rbind(emp_rate_25_to_34_df1, 
                              emp_rate_25_to_34_df2, 
                              emp_rate_25_to_34_df3, 
                              emp_rate_25_to_34_df4)


# 35-44 (Men, Women) (seasonally adjusted)
emp_rate_35_to_44_df1 <- bls_api(c("LNS12300173", "LNS12300334"), 
                                startyear = 1948, endyear = 1967,
                                registrationKey = Sys.getenv("BLS_KEY")) %>%
        spread(seriesID, value) %>% dateCast()
emp_rate_35_to_44_df2 <- bls_api(c("LNS12300173", "LNS12300334"), 
                                startyear = 1968, endyear = 1987,
                                registrationKey = Sys.getenv("BLS_KEY")) %>%
        spread(seriesID, value) %>% dateCast()
emp_rate_35_to_44_df3 <- bls_api(c("LNS12300173", "LNS12300334"), 
                                startyear = 1988, endyear = 2007,
                                registrationKey = Sys.getenv("BLS_KEY")) %>%
        spread(seriesID, value) %>% dateCast()
emp_rate_35_to_44_df4 <- bls_api(c("LNS12300173", "LNS12300334"), 
                                startyear = 2008, endyear = 2022,
                                registrationKey = Sys.getenv("BLS_KEY")) %>%
        spread(seriesID, value) %>% dateCast() %>% select(-latest)
emp_rate_35_to_44_df <- rbind(emp_rate_35_to_44_df1,
                              emp_rate_35_to_44_df2,
                              emp_rate_35_to_44_df3,
                              emp_rate_35_to_44_df4)


# 45-54 (Men, Women) (seasonally adjusted)
emp_rate_45_to_54_df1 <- bls_api(c("LNS12300182", "LNS12300341"), 
                                startyear = 1948, endyear = 1967,
                                registrationKey = Sys.getenv("BLS_KEY")) %>%
        spread(seriesID, value) %>% dateCast()
emp_rate_45_to_54_df2 <- bls_api(c("LNS12300182", "LNS12300341"), 
                                startyear = 1968, endyear = 1987,
                                registrationKey = Sys.getenv("BLS_KEY")) %>%
        spread(seriesID, value) %>% dateCast()
emp_rate_45_to_54_df3 <- bls_api(c("LNS12300182", "LNS12300341"), 
                                startyear = 1988, endyear = 2007,
                                registrationKey = Sys.getenv("BLS_KEY")) %>%
        spread(seriesID, value) %>% dateCast()
emp_rate_45_to_54_df4 <- bls_api(c("LNS12300182", "LNS12300341"), 
                                startyear = 2008, endyear = 2022,
                                registrationKey = Sys.getenv("BLS_KEY")) %>%
        spread(seriesID, value) %>% dateCast() %>% select(-latest)
emp_rate_45_to_54_df <- rbind(emp_rate_45_to_54_df1,
                              emp_rate_45_to_54_df2,
                              emp_rate_45_to_54_df3,
                              emp_rate_45_to_54_df4)


# 55+ (Men, Women) (seasonally adjusted)
emp_rate_55_df1 <- bls_api(c("LNS12324231", "LNS12324232"), 
                                startyear = 1948, endyear = 1967,
                                registrationKey = Sys.getenv("BLS_KEY")) %>%
        spread(seriesID, value) %>% dateCast()
emp_rate_55_df2 <- bls_api(c("LNS12324231", "LNS12324232"), 
                          startyear = 1968, endyear = 1987,
                          registrationKey = Sys.getenv("BLS_KEY")) %>%
        spread(seriesID, value) %>% dateCast()
emp_rate_55_df3 <- bls_api(c("LNS12324231", "LNS12324232"), 
                          startyear = 1988, endyear = 2007,
                          registrationKey = Sys.getenv("BLS_KEY")) %>%
        spread(seriesID, value) %>% dateCast()
emp_rate_55_df4 <- bls_api(c("LNS12324231", "LNS12324232"), 
                          startyear = 2008, endyear = 2022,
                          registrationKey = Sys.getenv("BLS_KEY")) %>%
        spread(seriesID, value) %>% dateCast() %>% select(-latest)
emp_rate_55_df <- rbind(emp_rate_55_df1,
                        emp_rate_55_df2,
                        emp_rate_55_df3,
                        emp_rate_55_df4)


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
