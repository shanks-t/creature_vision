provider "google" {
  project = "creature-vision"
  region  = var.region
}

terraform {
  backend "gcs" {
    bucket = "creature-vision-tf-state"
    prefix = "terraform/state"
  }
}
