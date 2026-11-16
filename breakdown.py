import sys
from question1.agent import SupportCopilot
from question2.agent import SupportPipeline

TICKETS = ["T-1001", "T-1002", "T-1003", "T-1004"]


def show(name, obj):
    prev = 0
    print(f"== {name} ==")
    for tid in TICKETS:
        obj.handle_ticket(tid)
        calls = obj.session.calls
        # every call made for this ticket since the last one
        total = obj.session.total_points()
        last = calls[-1]
        unc = last.input_tokens - min(last.cached_prefix_tokens, last.input_tokens)
        print(
            f"{tid}: points={total - prev:5d} | calls={last.label} | input={last.input_tokens} "
            f"(cached {last.cached_prefix_tokens}, uncached {unc}) | output={last.output_tokens}"
        )
        prev = total
    print("total", obj.session.total_points(), "\n")


show("Question 1", SupportCopilot())
show("Question 2", SupportPipeline())