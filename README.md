# mcp_hackathon

# Data Generation Server for ICD Code Finetuning

This MCP server provides tools for finetuning models to improve medical ICD-10 code prediction accuracy.

## Overview

The Data Generation Server is a Model Control Protocol (MCP) server that facilitates:

1. **Model Probing**: Evaluating model performance on ICD-10 code prediction
2. **Synthetic Data Generation**: Creating training data for model finetuning

## Set up the server

```
{
  "mcpServers": {
      "data_gen_server": {
          "command": "<base_dir>/.local/bin/uv",
          "args": [
              "--directory",
              "<base_dir>/mcp_hackathon/data_gen_server",
              "run",
              "data_gen_server.py"
          ]
      }
  }
}
```
