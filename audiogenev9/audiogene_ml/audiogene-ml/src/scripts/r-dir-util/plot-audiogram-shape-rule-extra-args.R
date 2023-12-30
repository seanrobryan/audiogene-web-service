# Title     : Audiogram-Shape Plot (Age Group)
# Objective : Plot Audioprofiles by Age Group w/ Rulings
# Created by: chichinwakama
# Created on: 16/6/2021
# Updated by: seanryan
# Updated on: 28/11/2022

# options(warn=-1)

library(dplyr)
library(tidyr)
library(ggplot2)
library(plotly)

# command args
args <- commandArgs(trailingOnly = TRUE)

if (length(args) )

# analysis parameters
rule_num <- args[1]
age_bin_size <- args[2]
threshold_1 <- args[3]
threshold_2 <- args[4]
data_file <- args[5]
output_dir <- args[6]
f1 <- args[7]
f2 <- args[8]
num_bins <- args[9]

rule_dir <- file.path(output_dir, paste("rule", rule_num, "nbins", num_bins, "size", age_bin_size, sep="_"))
analysis_dir <- file.path(rule_dir, paste(threshold_1, threshold_2, sep="_"))

# Create directory for generate files named by threshold values used in analysis
dir.create(analysis_dir, recursive = TRUE)

t <- paste0("Rule ", rule_num, " - Thresholds: ", f1, "=", threshold_1, ", ", f2, "=", threshold_2)

file1 <- file.path(analysis_dir, paste("shape_audioprofiles_rule", rule_num, "individual.html", sep="_"))
file2 <- file.path(analysis_dir, paste("shape_audioprofiles_rule", rule_num, "majority.html", sep="_"))
file3 <- file.path(analysis_dir, paste("rule", rule_num, "gene_shape_by_bin.csv", sep="_"))
cookie_bite_threshold <- 10
fun <- min # max mean median
fun2 <- max

fun <- switch(
  f1,
  "min" = min,
  "max" = max,
  "mean" = mean,
  "median" = median,
)

fun2 <- switch(
  f2,
  "min" = min,
  "max" = max,
  "mean" = mean,
  "median" = median,
)

df <- read.csv(data_file)

col_freq <- c("X125", "X250", "X500", "X1000", "X1500", "X2000", "X3000", "X4000", "X6000", "X8000")
low <- col_freq[1:3]
high <- col_freq[8:10]
medium <- col_freq[4:7]

# Age Bin - 20
age_group_f_20_2_bin <- function (df) {
  df_shape_age_group <- df %>% mutate(age_group = case_when(age < 20 ~ "0-20", TRUE ~ "20+")) %>%
    select(-c('age', 'id')) %>%
    add_count(locus, age_group) %>%
    group_by(locus, n, age_group) %>% summarise_at(vars(contains("dB")), list(~ mean(.))) %>%
    rename_with(~gsub(".dB", "", .x))  %>% ungroup()

  df_shape_age_group
}

# Age Bin - 30
age_group_f_30_2_bin <- function (df) {
  df_shape_age_group <- df %>%
  mutate(age_group = case_when(age < 30 ~ "0-30", TRUE ~ "30+")) %>%
  select(-c('age', 'id')) %>%
  add_count(locus, age_group) %>%
  group_by(locus, n, age_group) %>%
  summarise_at(vars(contains("dB")), list(~ mean(.))) %>%
  rename_with(~gsub(".dB", "", .x)) %>%
  ungroup()

  df_shape_age_group
}

# Age Bin - 20
age_group_f_20 <- function (df) {
  df_shape_age_group <- df %>% mutate(age_group = case_when(age < 20 ~ "0-20",
       age >= 20 & age < 40 ~ "20-40",
       age >= 40 & age < 60 ~ "40-60", TRUE ~ "60+")) %>%
    select(-c('age', 'id')) %>%
    add_count(locus, age_group) %>%
    group_by(locus, n, age_group) %>% summarise_at(vars(contains("dB")), list(~ mean(.))) %>%
    rename_with(~gsub(".dB", "", .x))  %>% ungroup()

  df_shape_age_group
}

# Age Bin - 30
age_group_f_30 <- function (df) {
  df_shape_age_group <- df %>%
  mutate(age_group = case_when(age < 30 ~ "0-30",
                               age >= 30 & age < 60 ~ "30-60", TRUE ~ "60+")) %>%
  select(-c('age', 'id')) %>%
  add_count(locus, age_group) %>%
  group_by(locus, n, age_group) %>%
  summarise_at(vars(contains("dB")), list(~ mean(.))) %>%
  rename_with(~gsub(".dB", "", .x)) %>%
  ungroup()

  df_shape_age_group
}

# TODO - Generalized Function (Not Important at the moment)

# adding counts for locus and summarising with the metric mean to produce audioprofiles by age group
bin_ages <- function (df, bin_size) {
  if (num_bins == 2) {
    if (bin_size == 20) {
      df_shape_age_group <- age_group_f_20_2_bin(df)
    } else if (bin_size == 30) {
      df_shape_age_group <- age_group_f_30_2_bin(df)
    }
  }
  else {
    if (bin_size == 20) {
    df_shape_age_group <- age_group_f_20(df)
  } else if (bin_size == 30) {
    df_shape_age_group <- age_group_f_30(df)
  }
  }
  df_shape_age_group
}

df_shape_age_group <- bin_ages(df, age_bin_size)

# adding counts for locus and summarising with the metric mean to produce audioprofiles
df_extra <- df %>%
  select(-c('age', 'id')) %>%
  add_count(locus) %>%
  mutate(age_group = 'everything') %>%
  group_by(locus, n, age_group) %>%
  summarise_at(vars(contains("dB")), list(~ mean(.))) %>%
  rename_with(~gsub(".dB", "", .x))

# combine age groups and overall audioprofile
df_shape <- rbind(df_shape_age_group, df_extra)

# Rules to decide shape which requires two data frames
# Rule #1 - Simple
rule_1 <- function (df, freq1 = `X250`, freq2 = `X6000`){
  df_shape <- df %>% group_by(locus, age_group) %>%
        mutate(shape = case_when(abs(freq1 - freq2) < threshold_1 ~ "cookie_bite",
            freq1 - freq2 < 0 ~ "downsloping", TRUE ~ "upsloping"))

  df_shape
}

rule_2 <- function(df_shape_age_group, low, medium, high, threshold = 25, threshold2 = 10) {
  df_shape_age_group['low'] <- apply(df_shape_age_group[, low], 1, fun)
  df_shape_age_group['high'] <- apply(df_shape_age_group[, high], 1, fun)
  df_shape_age_group['medium'] <- apply(df_shape_age_group[, medium], 1, fun2)

  df_shape <- df_shape_age_group %>%
        group_by(locus, age_group) %>%
        mutate(shape = case_when((abs(low - high) < threshold) &
                                 (medium - low > threshold2 & medium - high > threshold2) ~ "cookie_bite",
            (low - high) < 0 ~ "downsloping", TRUE ~ "upsloping"))

  df_shape
}

# Rule #2.5 – Flat
rule_2_5 <- function(df_shape_age_group, low, medium, high, threshold = 25, theshold2 = 10, fun = median, fun2 = max) {
  df_shape_age_group['low'] <- apply(df_shape_age_group[, low], 1, fun)
  df_shape_age_group['high'] <- apply(df_shape_age_group[, high], 1, fun)
  df_shape_age_group['medium'] <- apply(df_shape_age_group[, medium], 1, fun2)

  df_shape <- df_shape_age_group %>% group_by(locus, age_group) %>%
    mutate(shape = case_when((abs(low - high) < cookie_bite_threshold) &
                             (abs(medium - low) < cookie_bite_threshold | abs(medium - high) < cookie_bite_threshold) ~ "flat",
                           (abs(low - high) < theshold) &
                             (medium - low > theshold2 & medium - high > theshold2) ~ "cookie_bite",
                           (low - high) < 0 ~ "downsloping", TRUE ~ "upsloping"))

  df_shape
}


rule_3 <- function(df_shape_age_group, low, medium, high, ratio_threshold = 1, ratio_step = 0.2) {

  df_shape_age_group['low'] <- apply(df_shape_age_group[, low], 1, fun)
  df_shape_age_group['high'] <- apply(df_shape_age_group[, high], 1, fun)
  df_shape_age_group['medium'] <- apply(df_shape_age_group[, medium], 1, fun)

  df_shape <- df_shape_age_group %>%
          group_by(locus, age_group) %>%
          mutate(shape = case_when(((low/high >= (ratio_threshold - ratio_step)) &
                                    (low/high <= (ratio_threshold + ratio_step))) &
                                   (low/medium < ratio_threshold | high/medium < ratio_threshold) ~ "cookie_bite",
              (low/high) > 1 ~ "upsloping", TRUE ~ "downsloping"))

  df_shape
}

# Rule 4 - Four-Five metric

## TODO Make these rules to Functions


# Configuring Shape
# df_shape <- rule_2_5(df_shape, low, medium, high, theshold, theshold2, fun, fun2)
if(rule_num == 1) {
  df_shape <- rule_1(df_shape)
} else if (rule_num == 2) {
  df_shape <- rule_2(df_shape, low, medium, high, threshold_1, threshold_2)
} else if (rule_num == 2.5) {
  df_shape <- rule_2_5(df_shape, low, medium, high, threshold_1, threshold_2, fun, fun2)
} else if (rule_num == 3) {
  df_shape <- rule_3(df_shape, low, medium, high)
} else {
  # TODO: Add more options later (as seen in the above TODO)
}

g <- df_shape %>%
  group_by(age_group, shape) %>%
  mutate(nn = sum(n)) %>%
  ungroup() %>%
   pivot_longer(cols = all_of(col_freq),
               names_to = "Frequency", values_to = "dB") %>% mutate(Frequency = gsub("X", "", Frequency)) %>%
  highlight_key(~locus, "locus") %>%
  ggplot(aes(x = as.factor(as.integer(Frequency)), y = dB, colour = locus, group = locus, count = n, percentage = nn)) +
  geom_line(size = 0.75) +
  scale_colour_discrete(labels = paste0(levels(as.factor(df_shape$locus)), " (n=", df_shape$n, ")")) +
  geom_point() +
  ggtitle(t) +
  theme(axis.text.x = element_text(angle = 90)) +
  labs(x = "Frequency (Hz)") +
  scale_y_continuous(name = 'Hearing Loss (dB)', trans = "reverse") +
  facet_grid(shape ~ age_group)

p <- ggplotly(g, tooltip = c("colour", 'count', 'y', 'percentage', dynamicTicks = TRUE)) %>%
  highlight(on = "plotly_click", off = "plotly_doubleclick", color = 'seagreen')

htmlwidgets::saveWidget(as_widget(p), file1)

# Save Rule Table
df_majority <- df_shape %>%
  ungroup() %>%
  select(c(locus, shape)) %>%
  group_by(locus) %>%
  count(shape) %>%
  filter(n == max(n))

df_shape_majority_rule <- df_shape %>%
  select(-'shape') %>%
  left_join(df_majority %>% select(-'n'), by = 'locus')

### Other Graph who majority rule
g <- df_shape_majority_rule %>%
  pivot_longer(cols = all_of(col_freq),
               names_to = "Frequency", values_to = "dB") %>%
  mutate(Frequency = gsub("X", "", Frequency)) %>%
  highlight_key(~locus, "locus") %>%
  ggplot(aes(x = as.factor(as.integer(Frequency)), y = dB, colour = locus, group = locus, count = n)) +
  geom_line(size = 0.75) +
  scale_colour_discrete(labels = paste0(levels(as.factor(df_shape$locus)), " (n=", df_shape$n, ")")) +
  geom_point() +
  ggtitle(t) +
  theme(axis.text.x = element_text(angle = 90)) +
  labs(x = "Frequency (Hz)") +
  scale_y_continuous(name = 'Hearing Loss (dB)', trans = "reverse") +
  facet_grid(shape ~ age_group)

p2 <- ggplotly(g, tooltip = c("colour", 'count', 'y', dynamicTicks = TRUE)) %>%
  highlight(on = "plotly_click", off = "plotly_doubleclick", color = 'seagreen')

htmlwidgets::saveWidget(as_widget(p2), file2)

# df_shape <- transform(df_shape,
#                       fun_1 = f1,
#                       threshold_1 = threshold_1,
#                       fun_2 = f2,
#                       threshold_2 = threshold_2)

# o <- df_shape %>%
#   select(c('locus', 'age_group', 'shape', 'fun_1', 'threshold_1', 'fun_2', 'threshold_2')) %>%
#   pivot_wider(names_from = age_group, values_from = shape) %>%
#   left_join(df_majority %>% select(-'n'), by = 'locus') %>%
#   write.csv(file3, row.names = FALSE)

o <- df_shape %>%
  select(c('locus', 'age_group', 'shape')) %>%
  pivot_wider(names_from = age_group, values_from = shape) %>%
  left_join(df_majority %>% select(-'n'), by = 'locus') %>%
  write.csv(file3, row.names = FALSE)
