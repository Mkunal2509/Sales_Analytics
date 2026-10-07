# Create a local, writable library path inside the app directory
user_lib <- file.path(getwd(), "R_libs")
if (!dir.exists(user_lib)) {
  dir.create(user_lib, recursive = TRUE)
}
.libPaths(c(user_lib, .libPaths()))

# Only attempt install if not already installed system-wide via r-cran-forecast
if (!requireNamespace("forecast", quietly = TRUE)) {
  install.packages("forecast", lib = user_lib, repos = "https://cloud.r-project.org")
}