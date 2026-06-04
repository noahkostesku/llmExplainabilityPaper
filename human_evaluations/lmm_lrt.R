library(lme4)
library(lmerTest)

df <- read.csv("stats_out/survey_long.csv")

df$cohort <- ifelse(df$cohort == "CRPS", "CRP", "NCRP")
df$llm <- ifelse(df$model == "Model_1", "Gemma 3", ifelse(df$model == "Model_2", "DeepSeek R1", "Gemini 2.5"))

df_means <- aggregate(score ~ participant_id + cohort + llm + metric, data = df, FUN = mean)
cat(sprintf("Participant-level means: %d rows, %d participants\n",nrow(df_means), length(unique(df_means$participant_id))))

df_means$cohort <- factor(df_means$cohort, levels = c("CRP", "NCRP"))
df_means$llm <- factor(df_means$llm, levels = c("Gemma 3", "DeepSeek R1", "Gemini 2.5"))
df_means$metric <- factor(df_means$metric)

# M0 - no cohort
m0 <- lmer(score ~ llm * metric + (1 | participant_id), data = df_means, REML = FALSE)

# M1 - cohort main effect + cohort x metric, but no cohort x LLM
m1 <- lmer(score ~ cohort + llm * metric + cohort:metric + (1 | participant_id), data = df_means, REML = FALSE)

# M2 - full 3-way
m2 <- lmer(score ~ cohort * llm * metric + (1 | participant_id),data = df_means, REML = FALSE)

# M0 vs M2
cat("Test 1: M0 vs M2 (all cohort terms)\n")
lrt1 <- anova(m0, m2)
print(lrt1)
cat("\n")

# M1 vs M2
cat("Test 2: M1 vs M2 (cohort x LLM block)\n")
lrt2 <- anova(m1, m2)
print(lrt2)
cat("\n")

# Wald test of cohort:llm:metric in M2
cat("=== Test 3: Type III Wald tests for M2 ===\n")
anova_m2 <- anova(m2, type = "III", ddf = "Satterthwaite")
print(anova_m2)
cat("\n")

lrt1_df <- as.data.frame(lrt1)
lrt1_df$comparison <- "M0 vs M2"
lrt2_df <- as.data.frame(lrt2)
lrt2_df$comparison <- "M1 vs M2"
lrt_out <- rbind(lrt1_df, lrt2_df)
write.csv(lrt_out, "stats_out/lrt_results.csv", row.names = TRUE)

anova_out <- as.data.frame(anova_m2)
write.csv(anova_out, "stats_out/lmm_type3_anova.csv", row.names = TRUE)

metrics <- c("UND","TRU","INS","SAT","CON","CVN","COM","USB")
cohorts <- c("CRP","NCRP")
llms <- c("Gemma 3","DeepSeek R1","Gemini 2.5")

rows <- list()
for (co in cohorts) {
  for (ll in llms) {
    for (me in metrics) {
      sub <- df_means[df_means$cohort == co & df_means$llm == ll & df_means$metric == me, "score"]
      rows[[length(rows)+1]] <- data.frame(
        cohort = co, llm = ll, metric = me,
        N = length(sub), Mean = mean(sub), SD = sd(sub)
      )
    }
  }
}
table9 <- do.call(rbind, rows)
write.csv(table9, "stats_out/table9_descriptive_stats.csv", row.names = FALSE)

