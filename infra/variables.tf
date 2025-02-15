variable "billing_account" {
  default = null
}

variable "region" {
  default = "us-east1"
}

variable "zone" {
  description = "GCP Zone"
  type        = string
  default     = "us-east1-a"
}

variable "project_id" {
  description = "GCP Project ID"
  type        = string
  default     = "creature-vision"
}

