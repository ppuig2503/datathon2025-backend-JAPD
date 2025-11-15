class AnswerInput(BaseModel):
    """Input schema for answer requests"""
    question: str


class AnswerResponse(BaseModel):
    """Output schema for answer responses"""
    answer: str