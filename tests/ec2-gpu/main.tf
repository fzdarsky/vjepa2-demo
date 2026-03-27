terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Find latest RHEL 9 AMI
data "aws_ami" "rhel9" {
  most_recent = true
  owners      = ["309956199498"] # Red Hat

  filter {
    name   = "name"
    values = ["RHEL-9.*_HVM-*-x86_64-*-Hourly2-GP3"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

resource "aws_security_group" "gpu_test" {
  name_prefix = "vjepa2-gpu-test-"
  description = "SSH access for V-JEPA2 GPU testing"

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "vjepa2-gpu-test"
  }
}

resource "aws_instance" "gpu_test" {
  ami                    = data.aws_ami.rhel9.id
  instance_type          = var.instance_type
  key_name               = var.key_name
  vpc_security_group_ids = [aws_security_group.gpu_test.id]

  root_block_device {
    volume_size = 100
    volume_type = "gp3"
  }

  user_data = templatefile("${path.module}/user-data.sh", {
    rh_registry_user = var.rh_registry_user
    rh_registry_pass = var.rh_registry_pass
    hf_model_id      = var.hf_model_id
  })

  tags = {
    Name = "vjepa2-gpu-test"
  }
}
