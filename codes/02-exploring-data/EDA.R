library(tidyverse)
library(ggplot2)
library(scales)

## NCES DATA
df <- read.csv('../../data/01-modified-data/degrees_(by_sex_and_by_field)_final.csv')
df <- df %>% group_by(Measure, sex) %>% summarize(count=sum(Value)) %>% filter(sex=='M' | sex=='F')
level_order <- c("Bachelor's", "Master's", "Doctor's")
ggplot(df, aes(fill=sex, x=factor(Measure, level=level_order), y=count)) + 
        geom_bar(position='fill', stat='identity') + 
        scale_y_continuous(expand = c(0,0)) +
        ggtitle("Frequency of Degrees by sex (2018-2019)") +
        xlab('Degree') + 
        ylab('Frequency') + 
        theme_classic()

df <- read.csv('../../data/01-modified-data/counts_(by_sex_and_by_field)_final.csv')
level_order <- c("Transportation and materials moving", "Engineering technologies", 
                 "Military technologies and applied sciences",  
                 "Computer and information sciences and support services", 
                 "Engineering", "Philosophy and religious studies", 
                 "Theology and religious vocations", "History", "Mathematics and statistics", 
                 "Physical sciences and science technologies",  "Communications technologies", 
                 "Business", "Architecture and related services", 
                 "Parks, recreation, leisure, fitness, and kinesiology", 
                 "Homeland security, law enforcement, and firefighting",
                 "Social sciences and history", "Social sciences", 
                 "Agriculture and natural resources", "Precision production", 
                 "Visual and performing arts", "Communication, journalism, and related programs", 
                 "Biological and biomedical sciences", 
                 "Liberal arts and sciences, general studies, and humanities", 
                 "Multi/interdisciplinary studies", "Foreign languages, literatures, and linguistics", 
                 "Legal professions and studies", "English language and literature/letters", 
                 "Area, ethnic, cultural, gender, and group studies", "Psychology", 
                 "Education", "Public administration and social services", 
                 "Health professions and related programs", "Family and consumer sciences/human sciences", 
                 "Library science", "All fields, total")
ggplot(df, aes(fill=sex, x=factor(Occupations, level=level_order), y=Value)) + 
        geom_bar(position='fill', stat='identity') + 
        scale_y_continuous(expand = c(0,0)) +
        coord_flip() +
        ggtitle("Frequency of Bachelor's degrees in specific fields of study by sex (2019-2020)") +
        xlab('Fields of Study') + 
        ylab('Frequency') + 
        theme_classic()

## BLS DIRECT DATA ##
df <- read.csv('../../data/01-modified-data/earnings_and_unemployment_rates_(by_educational_attainment)_final.csv')
ggplot(df[1:8,], aes(x=reorder(Educational.attainment, Value), y=Value)) + 
        geom_bar(stat='identity', position=position_dodge(), alpha=0.75, fill='darkgreen') + 
        scale_y_continuous(expand = c(0,0)) +
        coord_flip() +
        geom_text(aes(label=Value), fontface="bold", vjust=0.5, hjust=1.5, color="white", size=4) +
        ggtitle("Median weekly earnings (USD) for people above age 25 by level of education (2021)") +
        xlab('Level of Education') + 
        ylab('Median Weekly Earning (USD)') + 
        theme_classic()
ggplot(df[9:16,], aes(x=reorder(Educational.attainment, -Value), y=Value)) + 
        geom_bar(stat='identity', position=position_dodge(), alpha=0.75, fill='red') + 
        scale_y_continuous(expand = c(0,0)) +
        coord_flip() +
        geom_text(aes(label=Value), fontface="bold", vjust=0.5, hjust=1.5, color="white", size=4) +
        ggtitle("Unemployment rate (%) for people above age 25 by level of education (2021)") +
        xlab('Level of Education') + 
        ylab('Unemployment Rate (%)') + 
        theme_classic()

df <- read.csv('../../data/01-modified-data/employment_(by_occupation_and_by_sex)_final.csv')
df <- df %>% filter(sex=='M' | sex=='F')
dfm <- df[df$sex=='M',]
dff <- df[df$sex=='F',]
level_order <- c("Healthcare support occupations" ,
                 "Personal care and service occupations", "Healthcare practitioners and technical occupations", 
                 "Education, training, and library occupations", "Office and administrative support occupations", 
                 "Community and social service occupations", "Production occupations", "Computer and mathematical occupations", 
                 "Production, transportation, and material moving occupations", "Farming, fishing, and forestry occupations", 
                 "Protective service occupations", "Transportation and material moving occupations", 
                 "Architecture and engineering occupations", "Natural resources, construction, and maintenance occupations", 
                 "Construction and extraction occupations", "Installation, maintenance, and repair occupations")
ggplot(df[abs(dfm$Value - dff$Value) > 33,], aes(fill=sex, x=factor(Occupation, level=level_order), y=Value)) +
        geom_bar(position='fill', stat='identity') +
        scale_y_continuous(expand = c(0,0)) +
        coord_flip() +
        ggtitle("Most polarized occupations for people above age 16 by sex (2021)") +
        xlab('Occupation') + 
        ylab('Frequency') + 
        theme_classic()

df <- read.csv('../../data/01-modified-data/employmentRates_final.csv')
ggplot(df, aes(x=date, y=Value, group=sex, color=sex)) +
        geom_line() +
        ggtitle("Employment Rates for various age groups in the U.S. (1948-2022)") +
        xlab('Date') + 
        ylab('Employment Rate (%)') + 
        scale_color_manual(values=c('darkgreen', 'red', 'blue')) +
        theme_classic() +
        theme(axis.ticks.x=element_blank(),
              axis.text.x=element_blank()) +
        facet_wrap(~Measure)

df <- read.csv('../../data/01-modified-data/hours_worked_(by_sex_and_by_occupation)_final.csv')
df1 <- df[df$Measure=="No. people at work" | df$Measure=='No. people who worked < 35hrs' | df$Measure=='No. people who worked 35+ hrs',] %>%
        filter(!row_number() %in% c(1, 7, 13, 19, 25, 31, 37, 43, 49, 55, 61, 67, 73, 79, 85)) %>% 
        mutate(Value = Value / 1000)
df2 <- df[df$Measure=="Average hrs worked among all workers" | df$Measure=='Average hrs worked among full-time workers',] %>%
        filter(!row_number() %in% c(1, 7, 13, 19, 25, 31, 37, 43, 49, 55, 61, 67, 73, 79, 85))
ggplot(df1[df1$sex=='M' | df1$sex=='F',], aes(fill=sex, x=Category, y=Value)) + 
        geom_bar(position='dodge', stat='identity') + 
        scale_y_continuous(expand = c(0,0)) +
        coord_flip() +
        ggtitle("Number of workers (in thousands) in various fields by sex (2021)") +
        xlab('Fields') + 
        ylab('Number of workers (in thousands)') + 
        theme_classic() +
        facet_wrap(~Measure)
ggplot(df2[df2$sex=='M' | df2$sex=='F',], aes(fill=sex, x=Category, y=Value)) + 
        geom_bar(position='dodge', stat='identity') + 
        scale_y_continuous(expand = c(0,0)) +
        coord_flip() +
        ggtitle("Hours worked in various fields by sex (2021)") +
        xlab('Fields') + 
        ylab('Hours worked') + 
        theme_classic() +
        facet_wrap(~Measure)
df2[df2$sex=='M',]$Value - df2[df2$sex=='F',]$Value

df <- read.csv('../../data/01-modified-data/wages_(by_occupation_may_2021)_final.csv')
ggplot(df[df$Measure=='Mean Annual Wage',], aes(fill=Occupation, x=Occupation, y=Value)) + 
        geom_bar(position='dodge', stat='identity') + 
        scale_y_continuous(expand = c(0,0)) +
        coord_flip() +
        ggtitle("Mean annual wage (USD) in various fields (May 2021)") +
        xlab('Fields') + 
        ylab('Mean Annual Wage (USD)') + 
        theme_classic() +
        theme(legend.position="none")
df[df$Measure=='Mean Annual Wage',] %>% arrange(desc(Value))

df <- read.csv('../../data/01-modified-data/occupations_detailed_(employment_and_wage).csv')
df <- df[df['O_GROUP']=='detailed',]
rownames(df) <- NULL # reset indices

service <- (sum(df[176:192,]$A_MEAN) + sum(df[201:265,]$A_MEAN) + sum(df[372:546,]$A_MEAN))/(length(df[176:192,]$A_MEAN) + length(df[372:546,]$A_MEAN) + length(df[201:265,]$A_MEAN))
labor <- mean(df[547:nrow(df),]$A_MEAN)
professional <- (sum(df[1:38,]$A_MEAN) + sum(df[71:175,]$A_MEAN) + sum(df[193:200,]$A_MEAN) + sum(df[301:371,]$A_MEAN))/(length(df[1:38,]$A_MEAN) + length(df[71:175,]$A_MEAN) + length(df[193:200,]$A_MEAN) + length(df[301:371,]$A_MEAN))
arts <- mean(df[265:299,]$A_MEAN)

## CAWP DATA ##
df <- read.csv('../../data/01-modified-data/percent_us_women_in_gov_final.csv')
ggplot(df, aes(x=Year, y=Value)) +
        geom_line() +
        ggtitle("Female representation (%) in the U.S. government (1965-2021)") +
        xlab('Year') + 
        ylab('Female Representation (%)') + 
        theme_classic() +
        facet_wrap(~Measure)
df[df$Year=='2021',]

## CB DATA ##
df <- read.csv('../../data/01-modified-data/Percentage-of-the-us-population-with-a-college-degree-by-gender-1940-2021.csv')
ggplot(df, aes(x=Year, y=Value, group=Measure, color=Measure)) +
        geom_line(size=1.5) +
        ggtitle("Percentage of college degree holders in the U.S. by sex (1940-2021)") +
        xlab('Date') + 
        ylab('Percentage of college degree holders (%)') + 
        scale_color_manual(values=c('blue', 'red')) +
        theme_classic()
df[df$Value[df$Measure=='Men with a college degree in the U.S. (%)'] < df$Value[df$Measure=='Women with a college degree in the U.S. (%)'],]

df <- read.csv('../../data/01-modified-data/School_Completion_(by_Age_and_Sex)_from_1940_to_2021_final.csv')
df$Value <- df$Value / 1000
ggplot(filter(df, grepl("0-4 years of education", Measure)), aes(x=Year, y=Value, group=sex, color=sex)) +
        geom_line() +
        ggtitle("Number of people (in thousands) by sex and by age group who have completed 0-4 years of education (1940-2021)") +
        xlab('Year') + 
        ylab('Number of people (in thousands)') + 
        scale_color_manual(values=c('red', 'blue')) +
        theme_classic() +
        facet_wrap(~Measure)
ggplot(filter(df, grepl("5-8 years of education", Measure)), aes(x=Year, y=Value, group=sex, color=sex)) +
        geom_line() +
        ggtitle("Number of people (in thousands) by sex and by age group who have completed 5-8 years of education (1940-2021)") +
        xlab('Year') + 
        ylab('Number of people (in thousands)') + 
        scale_color_manual(values=c('red', 'blue')) +
        theme_classic() +
        facet_wrap(~Measure)
ggplot(filter(df, grepl("9-11 years of education", Measure)), aes(x=Year, y=Value, group=sex, color=sex)) +
        geom_line() +
        ggtitle("Number of people (in thousands) by sex and by age group who have completed 9-11 years of education (1940-2021)") +
        xlab('Year') + 
        ylab('Number of people (in thousands)') + 
        scale_color_manual(values=c('red', 'blue')) +
        theme_classic() +
        facet_wrap(~Measure)
ggplot(filter(df, grepl("12 years of education", Measure)), aes(x=Year, y=Value, group=sex, color=sex)) +
        geom_line() +
        ggtitle("Number of people (in thousands) by sex and by age group who have completed 12 years of education (1940-2021)") +
        xlab('Year') + 
        ylab('Number of people (in thousands)') + 
        scale_color_manual(values=c('red', 'blue')) +
        theme_classic() +
        facet_wrap(~Measure)
ggplot(filter(df, grepl("13-15 years of education", Measure)), aes(x=Year, y=Value, group=sex, color=sex)) +
        geom_line() +
        ggtitle("Number of people (in thousands) by sex and by age group who have completed 13-15 years of education (1940-2021)") +
        xlab('Year') + 
        ylab('Number of people (in thousands)') + 
        scale_color_manual(values=c('red', 'blue')) +
        theme_classic() +
        facet_wrap(~Measure)
ggplot(filter(df, grepl("16", Measure)), aes(x=Year, y=Value, group=sex, color=sex)) +
        geom_line() +
        ggtitle("Number of people (in thousands) by sex and by age group who have completed 16+ years of education (1940-2021)") +
        xlab('Year') + 
        ylab('Number of people (in thousands)') + 
        scale_color_manual(values=c('red', 'blue')) +
        theme_classic() +
        facet_wrap(~Measure)