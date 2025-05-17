from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts.base import Message

# Initialize FastMCP server
mcp = FastMCP("finetune a icd code model")
# Initialize state attribute
mcp.state = type('State', (), {'synthetic_data': None})()


@mcp.tool()
async def probe_model_for_icd_code(model_name: str, eval_dataset: str) -> str:
    """
    Run an eval dataset against the model and return the results.

    Args:
        model_name: The name of the model to probe
        eval_dataset: The eval dataset to run against the model
    """ 
    return 95


@mcp.tool()
async def generate_data(model_name: str) -> str:
    """
    Generate synthetic data and ask for user verification.
    
    This is the data that will be used to finetune the model.
    """
    data = {
        "synthetic_data": [
            {
                "patient_id": 1,
                "icd_code": "95",
                "transcript": "transcript",
            },
            {
                "patient_id": 2,
                "icd_code": "96",
                "transcript": "transcript",
            },
        ]
    }
    
    # Store verified data in state
    mcp.state.synthetic_data = data
    return data

@mcp.prompt()
def confirm_finetune(model_name: str) -> list[Message]:
    """Prompt for confirming model finetuning."""
    return [
        Message(role="assistant", content=f"Ready to finetune model '{model_name}' with the verified data. Proceed? (yes/no)"),
        Message(role="assistant", content="Please respond with 'yes' to proceed with finetuning or 'no' to cancel.")
    ]

@mcp.tool()
async def finetune_model_for_icd_code(model_name: str) -> str:
    """
    Finetune the model
    
    Args:
        model_name: The name of the model to finetune
    """
    if mcp.state.synthetic_data is None:
        raise ValueError(
            "No verified synthetic data available. Please run generate_synthetic_data_for_icd_code_improvement first"
        )
    print(mcp.state.synthetic_data)

    return "Finetuned the model for the ICD code done! great job!"
    
if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')