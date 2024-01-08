# Title     : Audiogram-Shape Plot (Age Group)
# Objective : Plot Audioprofiles by Age Group
# Created by: chichinwakama
# Created on: 14/6/2021

options(warn=-1)

library(dplyr)
library(tidyr)
library(ggplot2)

# command args
args <- commandArgs(trailingOnly = TRUE)

# file parameters
file1 <- args[1]
file2 <- args[2]
file3 <- args[3]

### Comments args examples
# file1 <- "../../master_interp_fixed_revise_remove_unknown_loci_name.csv"
# file2 <- "../gene_annotatated.csv"
# file3 <- "../model_results_restore/shape_audioprofiles_test.html"

df <- read.csv(file1)
annot <- read.csv(file2)

df_count_merge <- df %>% left_join(annot, by='locus')
df_age_long <- df_count_merge %>%
  mutate(age_group = case_when(age < 20 ~ "0-20",
                               age >= 20 & age < 40 ~ "20-40",
                               age >= 40 & age < 60 ~ "40-60", TRUE ~ "60+")) %>%
  select(-c('age', 'id')) %>%
  add_count(locus, age_group) %>%
  group_by(locus, n, age_group) %>%
  summarise_at(vars(contains("dB")), list(~ mean(.)))

df_age <- df_age_long %>%
  pivot_longer(cols = contains("dB"),  names_to = "Frequency", values_to = "dB") %>%
  mutate(Frequency = as.integer(gsub(".dB", "", Frequency))) %>%
  ungroup()

df_age_long <- df_age_long %>% rename_with(~gsub(".dB", "", .x))
df_shape <- df_age_long %>% left_join(annot, by = 'locus')

df_extra <- df %>% select(-c('age', 'id')) %>%
  add_count(locus) %>%
  mutate(age_group = 'everything') %>%
  group_by(locus, n, age_group) %>% summarise_at(vars(contains("dB")), list(~ mean(.))) %>%
  rename_with(~gsub(".dB", "", .x)) %>%
  left_join(annot, by = 'locus')

df_shape <- rbind(df_shape, df_extra)

g <- df_shape %>% group_by(age_group, shape) %>%
  mutate(nn = sum(n)) %>%
  ungroup() %>%
  pivot_longer(cols = c('X125', 'X250', 'X500', 'X1000', 'X1500', 'X2000', 'X3000', 'X4000', 'X6000', 'X8000'),
               names_to = "Frequency", values_to = "dB") %>% mutate(Frequency = gsub("X", "", Frequency)) %>%
  plotly::highlight_key(~locus, "locus") %>%
  ggplot(aes(x = as.factor(as.integer(Frequency)), y = dB, colour = locus, group = locus, count = n, other = nn)) +
  geom_line(size = 0.75) +
  scale_colour_discrete(labels = paste(levels(as.factor(df_shape$locus)), " (n=", df_shape$n, ")", sep = "")) +
  geom_point() +
  ggtitle("") +
  theme(axis.text.x = element_text(angle = 90)) +
  labs(x = "Frequency (Hz)") +
  scale_y_continuous(name = 'Hearing Loss (dB)', trans = "reverse") +
  facet_grid(shape ~ age_group)


p <- plotly::ggplotly(g, tooltip=c("colour", 'count', 'other')) %>%
  plotly::highlight(on="plotly_click", off="plotly_doubleclick", color='seagreen')

htmlwidgets::saveWidget(plotly::as_widget(p), file3)

print(file3)