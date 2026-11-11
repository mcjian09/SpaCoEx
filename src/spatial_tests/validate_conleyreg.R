# Rscript validate_conleyreg.R [folder containing spot_outcomes.csv and cutoff_sensitivity.csv]
# Requires the CRAN package conleyreg. Does not rerun SpatialCorr or edit the original results.
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1])
args <- commandArgs(trailingOnly = TRUE)
folder <- if (length(args)) args[1] else
    file.path(dirname(normalizePath(script)), "B2M_HLA-A_SpatialCorr_Conley_log1p")
check_lib <- file.path(Sys.getenv("TEMP"), "spacoex_conleyreg_check_46")
if (dir.exists(check_lib)) .libPaths(c(check_lib, .libPaths()))
stopifnot(requireNamespace("conleyreg", quietly = TRUE))
obs <- read.csv(file.path(folder, "spot_outcomes.csv"), check.names = FALSE)
py <- read.csv(file.path(folder, "cutoff_sensitivity.csv"), check.names = FALSE)
labels <- tolower(as.character(obs$cancer))
stopifnot(all(labels %in% c("true", "false")), !anyDuplicated(obs$spot))
group <- as.integer(labels == "true")
stopifnot(all(py$cancer_n == sum(group)), all(py$noncancer_n == sum(1 - group)))
n <- nrow(obs)
# A supplied distance matrix makes dist_cutoff use its units, not geographic km.
D <- as.matrix(dist(as.matrix(obs[c("pixel_col", "pixel_row")])))
stopifnot(n == 3797L, nrow(py) == 18L, all(diag(D) == 0))
# conleyreg 0.1.9 strips a units attribute from externally supplied matrices.
# Give it a dimensionless attribute; the numbers remain pixel distances.
D <- units::set_units(D, "1", mode = "standard")
hc1 <- sqrt(n / (n - 2L))  # An intercept and one cancer indicator in Python
res <- do.call(rbind, lapply(seq_len(nrow(py)), function(i) {
    ref <- py[i, ]
    dat <- data.frame(y = obs[[ref$outcome]], cancer = group)
    out <- conleyreg::conleyreg(y ~ cancer, data = dat, model = "ols", kernel = "bartlett",
        dist_cutoff = ref$cutoff_px, dist_mat = D, ncores = 1L, verbose = FALSE)
    beta <- unname(out["cancer", 1]); se <- unname(out["cancer", 2])
    se_hc1 <- hc1 * se
    data.frame(outcome = ref$outcome, cutoff_px = ref$cutoff_px,
        python_difference = ref$difference, r_difference = beta,
        python_SE = ref$spatial_SE, r_package_SE = se, r_SE_with_python_correction = se_hc1,
        python_P = ref$P, r_package_P = unname(out["cancer", 4]),
        r_P_with_python_conventions = 2 * pnorm(-abs(beta / se_hc1)))
}))
print(packageVersion("conleyreg"))
print(res[res$cutoff_px == 2000, ], digits = 10, row.names = FALSE)
cat("May compare small p-values by relative, not absolute, error.\n")
cat("Max absolute difference in estimates:", max(abs(res$python_difference - res$r_difference)), "\n")
cat("Max absolute difference in SEs, after matching corrections:",
    max(abs(res$python_SE - res$r_SE_with_python_correction)), "\n")
cat("Changes in significance at 0.05 with R package defaults:",
    sum((res$python_P < .05) != (res$r_package_P < .05)), "/", nrow(res), "\n")
write.csv(res, file.path(folder, "validation_against_conleyreg.csv"), row.names = FALSE)
stopifnot(max(abs(res$python_difference - res$r_difference)) < 1e-9,
          max(abs(res$python_SE - res$r_SE_with_python_correction)) < 1e-9,
          max(abs(log(res$python_P) - log(res$r_P_with_python_conventions))) < 1e-7)
