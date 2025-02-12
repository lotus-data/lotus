import lotus
from lotus import parse_pdf
from lotus.models import LM, LiteLLMRM
import pathlib
from lotus.types import CascadeArgs, ProxyModel

gpt_4o_mini = LM("gpt-4o-mini")
gpt_4o = LM("gpt-4o")
rm = LiteLLMRM(model="text-embedding-3-small")

lotus.settings.configure(lm=gpt_4o, helper_lm=gpt_4o_mini, rm=rm)

# Load the PDF file
pdf_path = pathlib.Path(__file__).parent / "Poems on Love and Life.pdf"
df = parse_pdf(pdf_path, per_page=True)

user_instruction = "give me all the poems where {content} is motivating"
cascade_args = CascadeArgs(
    recall_target=0.9,
    precision_target=0.9,
    sampling_percentage=0.5,
    failure_probability=0.2,
    proxy_model=ProxyModel.HELPER_LM,
)

filtered_df = df.sem_filter(user_instruction=user_instruction, cascade_args=cascade_args)
top_motivating_poems = filtered_df.sem_topk("Which {content} is the most motivating?", K=1)

print(top_motivating_poems["content"].values[0])