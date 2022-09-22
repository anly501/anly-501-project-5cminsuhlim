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
# removing all-encompassing major titles
df <- df %>% filter(!row_number() %in% c(1:4, 38, 68, 69, 88, 113, 141, 158, 
                                         165, 178, 210, 257, 258, 273, 294, 
                                         307, 316, 337, 338, 357, 412, 413, 
                                         422, 460, 497, 498, 562))
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
df <- df[df$O_GROUP=='detailed',] %>% 
        select(-c(O_GROUP)) %>%
        mutate(TOT_EMP = as.numeric(TOT_EMP), 
               EMP_PRSE = as.numeric(EMP_PRSE),
               A_MEAN = as.numeric(A_MEAN), 
               MEAN_PRSE = as.numeric(MEAN_PRSE)) %>%
        rename(Occupation = OCC_TITLE, `Total Employment` = TOT_EMP,
               `Employment % Relative Standard Error` = EMP_PRSE,
               `Mean Annual Wage` = A_MEAN, 
               `Mean Annual Wage % Relative Standard Error` = MEAN_PRSE) %>%
        drop_na() ## removing na since empty data in this case is useless
write.csv(df, "wages_(by_occupation_may_2021)_final.csv", row.names=F)

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
write.csv(df, "percent_us_women_governors_final.csv", row.names=F)


df <- read.xlsx('../../data/00-raw-data/percent_us_women_house_rep.xlsx', 1, header=F)
df <- df %>% filter(row_number() %in% c(3:32))
names(df) <- df[1,]
df <- df[-1,] %>% 
        mutate(`Starting date of congressional term` = as.numeric(`Starting date of congressional term`),
               `Share of U.S. representatives who are women` = as.numeric(`Share of U.S. representatives who are women`) * 100) %>%
        rename(Year = `Starting date of congressional term`,
               `Female U.S. representatives (%)` = `Share of U.S. representatives who are women`)
write.csv(df, "percent_us_women_house_rep_final.csv", row.names=F)


df <- read.xlsx('../../data/00-raw-data/percent_us_women_senators.xlsx', 1, header=F)
df <- df %>% filter(row_number() %in% c(3:32))
names(df) <- df[1,]
df <- df[-1,] %>% 
        mutate(`Starting date of congressional term` = as.numeric(`Starting date of congressional term`),
               `Share of U.S. senators who are women` = as.numeric(`Share of U.S. senators who are women`) * 100) %>%
        rename(Year = `Starting date of congressional term`,
               `Female U.S. senators (%)` = `Share of U.S. senators who are women`) %>%
        select(-`NA`)
write.csv(df, "percent_us_women_senators_final.csv", row.names=F)


df <- read.xlsx('../../data/00-raw-data/percent_us_women_state_legislators.xlsx', 1, header=F)
df <- df %>% filter(row_number() %in% c(3:41))
names(df) <- df[1,]
df <- df[-1,] %>% 
        mutate(Year = as.numeric(Year),
               `Share of state legislators who are women` = as.numeric(`Share of state legislators who are women`) * 100) %>%
        rename(`Female U.S. state legislators (%)` = `Share of state legislators who are women`)
write.csv(df, "percent_us_women_state_legislators_final.csv", row.names=F)


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
write.csv(df, "percent_us_women_state_legislators_final.csv", row.names=F)


df <- read.xlsx('../../data/00-raw-data/School_Completion_(by_Age_and_Sex)_from_1940_to_2021.xlsx', 1, header=F)
df[7,1] <- "Year"
df[7,2] <- "Total"
df[7,3] <- "0-4 years"
df[7,4] <- "5-8 years"
df[7,5] <- "9-11 years"
df[7,6] <- "12 years"
df[7,7] <- "13-15 years"
df[7,8] <- "16+ years"
df <- df %>% select(c(X1:X8)) %>% filter(!row_number() %in% c(1:6))
names(df) <- df[1,]
df <- df[-1,]
df1 <- df %>% filter(row_number() %in% c(268:401))
df1_m <- df1 %>% filter(row_number() %in% c(2:67)) %>%
        rename(`Total (25-34, M)` = Total, 
               `0-4 years (25-34, M)` = `0-4 years`,
               `5-8 years (25-34, M)` = `5-8 years`,
               `9-11 years (25-34, M)` = `9-11 years`,
               `12 years (25-34, M)` = `12 years`,
               `13-15 years (25-34, M)` = `13-15 years`,
               `16+ years (25-34, M)` = `16+ years`)
df1_f <- df1 %>% filter(row_number() %in% c(69:134)) %>%
        rename(`Total (25-34, F)` = Total, 
               `0-4 years (25-34, F)` = `0-4 years`,
               `5-8 years (25-34, F)` = `5-8 years`,
               `9-11 years (25-34, F)` = `9-11 years`,
               `12 years (25-34, F)` = `12 years`,
               `13-15 years (25-34, F)` = `13-15 years`,
               `16+ years (25-34, F)` = `16+ years`)
df2 <- df %>% filter(row_number() %in% c(469:602))
df2_m <- df2 %>% filter(row_number() %in% c(2:67)) %>%
        rename(`Total (35-54, M)` = Total, 
               `0-4 years (35-54, M)` = `0-4 years`,
               `5-8 years (35-54, M)` = `5-8 years`,
               `9-11 years (35-54, M)` = `9-11 years`,
               `12 years (35-54, M)` = `12 years`,
               `13-15 years (35-54, M)` = `13-15 years`,
               `16+ years (35-54, M)` = `16+ years`)
df2_f <- df2 %>% filter(row_number() %in% c(69:134)) %>%
        rename(`Total (35-54, F)` = Total, 
               `0-4 years (35-54, F)` = `0-4 years`,
               `5-8 years (35-54, F)` = `5-8 years`,
               `9-11 years (35-54, F)` = `9-11 years`,
               `12 years (35-54, F)` = `12 years`,
               `13-15 years (35-54, F)` = `13-15 years`,
               `16+ years (35-54, F)` = `16+ years`)
df3 <- df %>% filter(row_number() %in% c(670:803))
df3_m <- df3 %>% filter(row_number() %in% c(2:67)) %>%
        rename(`Total (55+, M)` = Total, 
               `0-4 years (55+, M)` = `0-4 years`,
               `5-8 years (55+, M)` = `5-8 years`,
               `9-11 years (55+, M)` = `9-11 years`,
               `12 years (55+, M)` = `12 years`,
               `13-15 years (55+, M)` = `13-15 years`,
               `16+ years (55+, M)` = `16+ years`)
df3_f <- df3 %>% filter(row_number() %in% c(69:134)) %>%
        rename(`Total (55+, F)` = Total, 
               `0-4 years (55+, F)` = `0-4 years`,
               `5-8 years (55+, F)` = `5-8 years`,
               `9-11 years (55+, F)` = `9-11 years`,
               `12 years (55+, F)` = `12 years`,
               `13-15 years (55+, F)` = `13-15 years`,
               `16+ years (55+, F)` = `16+ years`)
df <- cbind(df1_m, df1_f[, 2:8], 
            df2_m[, 2:8], df2_f[, 2:8], 
            df3_m[, 2:8], df3_f[, 2:8]) %>% 
        relocate(c(9, 16, 23, 30, 37), .after=2) %>%
        relocate(c(14, 20, 26, 32, 38), .after=8) %>%
        relocate(c(19, 24, 29, 34, 39), .after=14) %>%
        relocate(c(24, 28, 32, 36, 40), .after=20) %>%
        relocate(c(29, 32, 35, 38, 41), .after=26) %>%
        relocate(c(34, 36, 38, 40, 42), .after=32) %>%
        relocate(c(39:43), .after=38)
df <- lapply(df, as.numeric)
write.csv(df, "School_Completion_(by_Age_and_Sex)_from_1940_to_2021_final.csv", row.names=F)


## NCES DIRECT DATA ##
df <- read.xlsx('../../data/00-raw-data/degrees_(by_sex_and_by_field).xls', 1, header=F)
df[3,2] <- "Occupations"
df[3,3] <- "Bachelor's Total"
df[3,4] <- "Bachelor's Men"
df[3,5] <- "Bachelor's Women"
df[3,6] <- "Master's Total"
df[3,7] <- "Master's Men"
df[3,8] <- "Master's Women"
df[3,9] <- "Doctor's Total"
df[3,10] <- "Doctor's Men"
df[3,11] <- "Doctor's Women"
# removing all-encompassing major titles
df <- df %>% select(-X1) %>%
        filter(!row_number() %in% c(1, 2, 4, 6, 7, 64, 65, 87, 98, 138,
                                    223, 314, 326, 327, 351, 365, 394, 
                                    492, 493, 547, 548, 607, 617, 629,
                                    642, 676, 730, 932, 961, 982, 987,
                                    992, 1008, 1020, 1053, 1066, 1081, 
                                    1125, 1133, 1138, 1166, 1179, 1180,
                                    1212, 1221, 1239, 1249, 1314:1317))
names(df) <- df[1,]
df <- df[-1,] %>% mutate(`Bachelor's Total` = as.numeric(`Bachelor's Total`),
                         `Bachelor's Men` = as.numeric(`Bachelor's Men`),
                         `Bachelor's Women` = as.numeric(`Bachelor's Women`),
                         `Master's Total` = as.numeric(`Master's Total`),
                         `Master's Men` = as.numeric(`Master's Men`),
                         `Master's Women` = as.numeric(`Master's Women`),
                         `Doctor's Total` = as.numeric(`Doctor's Total`),
                         `Doctor's Men` = as.numeric(`Doctor's Men`),
                         `Doctor's Women` = as.numeric(`Doctor's Women`))
df$Occupations <- str_trim(df$Occupations)
write.csv(df, "degrees_(by_sex_and_by_field)_final.csv", row.names=F)


df1 <- read.xlsx('../../data/00-raw-data/degrees_(by_field_males).xls', 1, header=F)
df1[5,1] <- "Occupations"
df1[5,2] <- "Total (M)"
df1 <- df1 %>% select(c(X1, X2)) %>%
        filter(row_number() %in% c(5:40))
names(df1) <- df1[1,]
df1 <- df1[-1,]
df1$Occupations <- gsub("\\s+", " ", str_trim(df1$Occupations))
df1$Occupations[2] <- substr(df1$Occupations[2], 1, nchar(df1$Occupations[2]) - 3)
df1$Occupations[6] <- substr(df1$Occupations[6], 1, nchar(df1$Occupations[6]) - 3)
df1$Occupations[12] <- substr(df1$Occupations[12], 1, nchar(df1$Occupations[12]) - 3)

df2 <- read.xlsx('../../data/00-raw-data/degrees_(by_field_females).xls', 1, header=F)
df2[5,1] <- "Occupations"
df2[5,2] <- "Total (F)"
df2 <- df2 %>% select(c(X1, X2)) %>%
        filter(row_number() %in% c(5:40))
names(df2) <- df2[1,]
df2 <- df2[-1,]
df2$Occupations <- gsub("\\s+", " ", str_trim(df2$Occupations))
df2$Occupations[2] <- substr(df2$Occupations[2], 1, nchar(df2$Occupations[2]) - 3)
df2$Occupations[6] <- substr(df2$Occupations[6], 1, nchar(df2$Occupations[6]) - 3)
df2$Occupations[12] <- substr(df2$Occupations[12], 1, nchar(df2$Occupations[12]) - 3)

df <- cbind(df1, df2[,2]) %>%
        rename(`Total (F)` = `df2[, 2]`) %>% 
        mutate(`Total (M)` = as.numeric(`Total (M)`),
               `Total (F)` = as.numeric(`Total (F)`))
write.csv(df, "degrees_(by_field_and_sex)_final.csv", row.names=F)


## BIG 5 API DATA ##
df <- read.csv('../../data/00-raw-data/big_five.csv', header=T)
df <- df %>% filter(country == 'USA') %>% arrange(sex)
write.csv(df, "big_five_final.csv", row.names=F)
