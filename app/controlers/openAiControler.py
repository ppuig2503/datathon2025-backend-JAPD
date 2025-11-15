from fastapi import APIRouter, HTTPException
from dotenv import load_dotenv
from openai import OpenAI
import os
from app.types.openAiTypes import AnswerInput, AnswerResponse

router = APIRouter(
    prefix="/ai",
    tags=["OpenAI"]
)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@router.post("/answer", response_model=AnswerResponse)
def answer(input_data: AnswerInput):
    """
    Answer endpoint using OpenAI API.
    
    Args:
        input_data: AnswerInput object with the question
        
    Returns:
        AnswerResponse with the answer from OpenAI
    """
    try:
        # TODO: Add system prompt and context as needed
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                # {"role": "system", "content": "Your system prompt here"},
                {"role": "user", "content": input_data.question}
            ]
        )
        
        answer = response.choices[0].message.content
        
        return AnswerResponse(answer=answer)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")