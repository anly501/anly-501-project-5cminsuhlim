library(tidyverse)
library(xlsx)

## BLS DIRECT DATA ##
df <- read.xlsx('../../data/00-raw-data/earnings_and_unemployment_rates_(by_educational_attaiment).xlsx', 1, header=F)
df <- df %>% filter(row_number() %in% c(2:10))
names(df) <- df[1,]
df <- df[-1,] %>%
        mutate(`Median usual weekly earnings` = as.numeric(`Median usual weekly earnings`),
               `Unemployment rate` = as.numeric(`Unemployment rate`)) %>%
        rename(`Median weekly earnings ($)` = `Median usual weekly earnings`,
               `Unemployment rate (%)` = `Unemployment rate`)
df[1, 3] <- df[1, 3] * 100 ## adjusting units for this item
write.csv(df, "earnings_and_unemployment_rates_(by_educational_attaiment)_final.csv", row.names=F)


## BLS API EMPLOYMENT RATE DATA ##
df <- read.csv('../../data/00-raw-data/employmentRate.csv')
df <- df %>% 
        rename(`Total Employment (%)` = LNS12300000, 
               `Employment among men (%)` = LNS12300001, 
               `Employment among women (%)` = LNS12300002) %>%
        mutate(period = as.numeric(str_sub(df$period, 2, 3))) %>%
        select(-footnotes)
write.csv(df, "employmentRate_final.csv", row.names=F)


df <- read.csv('../../data/00-raw-data/employmentRate16to24.csv')
df <- df %>% 
        rename(`Employment among men 16-24 (%)` = LNS12324885, 
               `Employment among women 16-24 (%)` = LNS12324886) %>%
        mutate(period = as.numeric(str_sub(df$period, 2, 3))) %>%
        select(-footnotes)
write.csv(df, "employmentRate16-24_final.csv", row.names=F)


df <- read.csv('../../data/00-raw-data/employmentRate25to34.csv')
df <- df %>% 
        rename(`Employment among men 25-34 (%)` = LNS12300164, 
               `Employment among women 25-34 (%)` = LNS12300327) %>%
        mutate(period = as.numeric(str_sub(df$period, 2, 3))) %>%
        select(-footnotes)
write.csv(df, "employmentRate25-34_final.csv", row.names=F)


df <- read.csv('../../data/00-raw-data/employmentRate35to44.csv')
df <- df %>% 
        rename(`Employment among men 35-44 (%)` = LNS12300173, 
               `Employment among women 35-44 (%)` = LNS12300334) %>%
        mutate(period = as.numeric(str_sub(df$period, 2, 3))) %>%
        select(-footnotes)
write.csv(df, "employmentRate35-44_final.csv", row.names=F)


df <- read.csv('../../data/00-raw-data/employmentRate45to54.csv')
df <- df %>% 
        rename(`Employment among men 45-54 (%)` = LNS12300182, 
               `Employment among women 45-54 (%)` = LNS12300341) %>%
        mutate(period = as.numeric(str_sub(df$period, 2, 3))) %>%
        select(-footnotes)
write.csv(df, "employmentRate45-54_final.csv", row.names=F)


df <- read.csv('../../data/00-raw-data/employmentRate55+.csv')
df <- df %>% 
        rename(`Employment among men 55+ (%)` = LNS12324231, 
               `Employment among women 55+ (%)` = LNS12324232) %>%
        mutate(period = as.numeric(str_sub(df$period, 2, 3))) %>%
        select(-footnotes)
write.csv(df, "employmentRate55+_final.csv", row.names=F)


## CAWP DIRECT DATA ##
df <- read.xlsx('../../data/00-raw-data/percent_us_women_governors.xlsx', 1, header=F)
df <- df %>% filter(row_number() %in% c(3:50))
names(df) <- df[1,]
df <- df[-1,] %>%
        mutate(Year = as.numeric(Year),
               `Share of state governors who are women` = as.numeric(`Share of state governors who are women`) * 100) %>%
        rename(`Female state governors (%)` = `Share of state governors who are women`)
write.csv(df, "percent_us_women_governors_final.xlsx", row.names=F)


df <- read.xlsx('../../data/00-raw-data/percent_us_women_house_rep.xlsx', 1, header=F)
df <- df %>% filter(row_number() %in% c(3:32))
names(df) <- df[1,]
df <- df[-1,] %>% 
        mutate(`Starting date of congressional term` = as.numeric(`Starting date of congressional term`),
               `Share of U.S. representatives who are women` = as.numeric(`Share of U.S. representatives who are women`) * 100) %>%
        rename(`Female U.S. representatives (%)` = `Share of U.S. representatives who are women`)
write.csv(df, "percent_us_women_house_rep_final.xlsx", row.names=F)


df <- read.xlsx('../../data/00-raw-data/percent_us_women_senators.xlsx', 1, header=F)
df <- read.xlsx('../../data/00-raw-data/percent_us_women_state_legislators.xlsx', 1, header=F)
