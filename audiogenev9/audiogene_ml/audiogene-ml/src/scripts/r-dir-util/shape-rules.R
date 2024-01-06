library(dplyr)
library(tidyr)

# Rules to decide shape which requires two data frames
# Rule #1 - Simple
rule_1 <- function (df, freq1 = `X250`, freq2 = `X6000`){
  df_shape <- df %>% group_by(locus, age_group) %>%
        mutate(shape = case_when(abs(freq1 - freq2) < threshold_1 ~ "cookie_bite",
            freq1 - freq2 < 0 ~ "downsloping", TRUE ~ "upsloping"))

  df_shape
}

rule_2 <- function(df_shape_age_group, low, medium, high, threshold = 25, threshold_2 = 10) {
  df_shape_age_group['low'] <- apply(df_shape_age_group[, low], 1, fun)
  df_shape_age_group['high'] <- apply(df_shape_age_group[, high], 1, fun)
  df_shape_age_group['medium'] <- apply(df_shape_age_group[, medium], 1, fun2)

  df_shape <- df_shape_age_group %>%
        group_by(locus, age_group) %>%
        mutate(shape = case_when((abs(low - high) < threshold) &
                                 (medium - low > threshold_2 & medium - high > threshold_2) ~ "cookie_bite",
            (low - high) < 0 ~ "downsloping", TRUE ~ "upsloping"))

  df_shape
}

# Rule #2.5 – Flat
rule_2_5 <- function(df_shape_age_group, low, medium, high, threshold = 25, threshold_2 = 10, cookie_bite_threshold = 10, fun = median, fun2 = max) {
  df_shape_age_group['low'] <- apply(df_shape_age_group[, low], 1, fun)
  df_shape_age_group['high'] <- apply(df_shape_age_group[, high], 1, fun)
  df_shape_age_group['medium'] <- apply(df_shape_age_group[, medium], 1, fun2)

  df_shape <- df_shape_age_group %>% group_by(locus, age_group) %>%
    mutate(shape = case_when((abs(low - high) < cookie_bite_threshold) &
                             (abs(medium - low) < cookie_bite_threshold | abs(medium - high) < cookie_bite_threshold) ~ "flat",
                           (abs(low - high) < threshold) &
                             (medium - low > threshold_2 & medium - high > threshold_2) ~ "cookie_bite",
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