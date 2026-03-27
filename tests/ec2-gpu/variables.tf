variable "aws_region" {
  description = "AWS region for the GPU test instance"
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "EC2 instance type (must have NVIDIA GPU)"
  type        = string
  default     = "g4dn.xlarge"
}

variable "key_name" {
  description = "Name of an existing EC2 key pair for SSH access"
  type        = string
}

variable "rh_registry_user" {
  description = "Red Hat registry username for pulling base images"
  type        = string
  sensitive   = true
}

variable "rh_registry_pass" {
  description = "Red Hat registry password/token"
  type        = string
  sensitive   = true
}

variable "hf_model_id" {
  description = "HuggingFace model ID to download for testing"
  type        = string
  default     = "facebook/vjepa2-vitl-fpc16-256-ssv2"
}

variable "allowed_ssh_cidr" {
  description = "CIDR block allowed to SSH into the instance"
  type        = string
  default     = "0.0.0.0/0"
}
