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

df <- read.xlsx('../../data/00-raw-data/employment_(by_occupation_and_by_sex).xlsx', 1, header=F)
df[7,1] <- "Occupation"
df[7,2] <- "Total Count"
df[7,3] <- "Women"
df[7,4] <- "Men"
df <- df %>% filter(row_number() %in% c(7:602)) %>% 
  select(c(X1, X2, X3, X4))
names(df) <- df[1,]
df <- df[-1,] %>%
        mutate(`Total Count` = as.numeric(`Total Count`) * 1000,
               Women = as.numeric(Women), Men = 100 - Women) %>%
        rename(`Women employed (%)` = Women, `Men employed (%)` = Men)
## keeping NAs, since they're located in subcategories
write.csv(df, "employment_(by_occupation_and_by_sex)_final.csv", row.names=F)


df <- read.xlsx('../../data/00-raw-data/hours_worked_(by_sex_and_by_occupation).xlsx', 1, header=F)
df <- df %>% filter(row_number() %in% c(9:50)) %>% 
        select(c(X1, X2, X3, X7, X8, X9)) %>%
        filter(row_number() %in% c(1, 2, 5, 6, 9, 12, 
                                   15, 16, 19, 20, 23, 26,
                                   29, 30, 33, 34, 37, 40)) %>% 
        mutate(X2 = as.numeric(X2) * 1000, X3 = as.numeric(X3) * 1000, 
               X7 = as.numeric(X7) * 1000, X8 = as.numeric(X8),
               X9 = as.numeric(X9))
df[1,1] <- 'Cumulative Counts'
df[7,1] <- 'Male Total'
df[13,1] <- 'Female Total'
df_t <- df[1:6,]
df_t <- df_t %>%
        rename(Category = X1, `No. people at work` = X2, 
               `No. people who worked < 35hrs` = X3, 
               `No. people who worked 35+ hrs` = X7, 
               `Average hrs worked among all workers` = X8, 
               `Average hrs worked among full-time workers` = X9)
df_m <- df[7:12,]
df_m <- df_m %>%
        rename(Category = X1, `No. men at work` = X2, 
               `No. men who worked < 35hrs` = X3, 
               `No. men who worked 35+ hrs` = X7, 
               `Average hrs worked among all men` = X8, 
               `Average hrs worked among full-time men` = X9)
df_f <- df[13:18,]
df_f <- df_f %>%
        rename(Category = X1, `No. women at work` = X2, 
               `No. women who worked < 35hrs` = X3, 
               `No. women who worked 35+ hrs` = X7, 
               `Average hrs worked among all women` = X8, 
               `Average hrs worked among full-time women` = X9)
df <- cbind(df_t, df_m[, 2:6], df_f[, 2:6])
df <- df %>% 
        relocate(`No. men at work`, .after=`No. people at work`) %>%
        relocate(`No. women at work`, .after=`No. men at work`) %>%
        relocate(`No. men who worked < 35hrs`, .after=`No. people who worked < 35hrs`) %>%
        relocate(`No. women who worked < 35hrs`, .after=`No. men who worked < 35hrs`) %>%
        relocate(`No. men who worked 35+ hrs`, .after=`No. people who worked 35+ hrs`) %>%
        relocate(`No. women who worked 35+ hrs`, .after=`No. men who worked 35+ hrs`) %>%
        relocate(`Average hrs worked among all men`, .after=`Average hrs worked among all workers`) %>%
        relocate(`Average hrs worked among all women`, .after=`Average hrs worked among all men`) %>%
        relocate(`Average hrs worked among full-time men`, .after=`Average hrs worked among full-time workers`) %>%
        relocate(`Average hrs worked among full-time women`, .after=`Average hrs worked among full-time men`)
write.csv(df, "hours_worked_(by_sex_and_by_occupation)_final.csv", row.names=F)


df <- read.xlsx('../../data/00-raw-data/wages_(by_occupation_may_2021).xlsx', 1, header=F)
df <- df[, c(10, 11, 12, 13, 19, 20)]
names(df) <- df[1,]
df <- df[-1,]

#Management, professional, and related occupations
#Service occupations
#Sales and office occupations
#Natural resources, construction, and maintenance occupations
#Production, transportation, and material moving occupations

## BLS EMPLOYMENT RATE DATA ##
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
        rename(Year = `Starting date of congressional term`,
               `Female U.S. representatives (%)` = `Share of U.S. representatives who are women`)
write.csv(df, "percent_us_women_house_rep_final.xlsx", row.names=F)


df <- read.xlsx('../../data/00-raw-data/percent_us_women_senators.xlsx', 1, header=F)
df <- df %>% filter(row_number() %in% c(3:32))
names(df) <- df[1,]
df <- df[-1,] %>% 
        mutate(`Starting date of congressional term` = as.numeric(`Starting date of congressional term`),
               `Share of U.S. senators who are women` = as.numeric(`Share of U.S. senators who are women`) * 100) %>%
        rename(Year = `Starting date of congressional term`,
               `Female U.S. senators (%)` = `Share of U.S. senators who are women`) %>%
        select(-`NA`)
write.csv(df, "percent_us_women_senators_final.xlsx", row.names=F)


df <- read.xlsx('../../data/00-raw-data/percent_us_women_state_legislators.xlsx', 1, header=F)
df <- df %>% filter(row_number() %in% c(3:41))
names(df) <- df[1,]
df <- df[-1,] %>% 
        mutate(Year = as.numeric(Year),
               `Share of state legislators who are women` = as.numeric(`Share of state legislators who are women`) * 100) %>%
        rename(`Female U.S. state legislators (%)` = `Share of state legislators who are women`)
write.csv(df, "percent_us_women_state_legislators_final.xlsx", row.names=F)


## CB DIRECT DATA ##
df <- read.xlsx('../../data/00-raw-data/Percentage-of-the-us-population-with-a-college-degree-by-gender-1940-2021.xlsx', 2, header=F)
df[3,1] <- "Year"
df[3,4] <- "drop"
df <- df %>% filter(row_number() %in% c(3:68))
names(df) <- df[1,]
df <- df[-1,] %>% 
        mutate(Year = as.numeric(Year), Male = as.numeric(Male), Female = as.numeric(Female)) %>%
        rename(`Men with a college degree in the U.S. (%)` = Male,
               `Women with a college degree in the U.S. (%)` = Female) %>%
        select(-drop)
write.csv(df, "percent_us_women_state_legislators_final.xlsx", row.names=F)


df <- read.xlsx('../../data/00-raw-data/School_Completion_(by_Age_and_Sex)_from_1940_to_2021.xlsx', 1, header=F)


## NCES DIRECT DATA ##
df <- read.xlsx('../../data/00-raw-data/degrees_(by_sex_and_by_field).xls', 1, header=F)
df <- read.xlsx('../../data/00-raw-data/degrees_(by_field_males).xls', 1, header=F)
df <- read.xlsx('../../data/00-raw-data/degrees_(by_field_females).xls', 1, header=F)