import argparse
import os
import io
import logging
import asyncio
import json
from PIL import Image
from langgraph.graph import StateGraph, END
from invoice_agents import AgentState, extract_node, audit_node, human_review_node


def setup_logger():
    logger = logging.getLogger()
    # Prevent adding duplicate handlers if function is called multiple times
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)

    # Format: [Time] [Level] Message
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Handler 1: Console (Terminal)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Handler 2: File (agent.log)
    fh = logging.FileHandler("agent.log", mode='w')  # mode='w' overwrites each run
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Silence noisy libraries that log at INFO level
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("googleapiclient").setLevel(logging.WARNING)

    return logger
setup_logger()
logger = logging.getLogger(__name__)


def compile_workflow():
    workflow = StateGraph(AgentState)
    workflow.add_node("extract", extract_node)
    workflow.add_node("audit", audit_node)
    workflow.add_node("human_review", human_review_node)

    workflow.set_entry_point("extract")
    workflow.add_edge("extract", "audit")

    def route_decision(state):
        return "human_review" if state["safety_check"] == "flagged" else END

    workflow.add_conditional_edges("audit", route_decision)
    return workflow.compile()


def load_images_generator(source):
    if os.path.isdir(source):
        files = [os.path.join(source, f) for f in os.listdir(source) if f.endswith(('.jpg', '.png'))]
    else:
        files = [source]

    logger.info(f"Found {len(files)} images in source.")

    for path in files:
        try:
            with Image.open(path) as img:
                img = img.convert('RGB')
                b = io.BytesIO()
                img.save(b, format='JPEG')
                yield os.path.basename(path), b.getvalue()
        except Exception as e:
            logger.error(f"Could not load {path}: {e}")


async def worker(app, semaphore, filename, img_bytes):
    async with semaphore:
        inputs = {"image_bytes": img_bytes, "image_path": filename, "messages": []}
        # Use ainvoke for async
        return await app.ainvoke(inputs)


async def main(app, input_folder, num_agents, test_mode=None):
    semaphore = asyncio.Semaphore(num_agents)
    tasks = []

    logger.info(f"Spawning Pool with {num_agents} Agents")
    if test_mode is not None:
        fname = test_mode[0]
        ibytes = test_mode[1]
        tasks.append(worker(app, semaphore, fname, ibytes))
    else:
        for fname, ibytes in load_images_generator(input_folder):
            tasks.append(worker(app, semaphore, fname, ibytes))

    # Run all tasks, ignore crashes
    results = await asyncio.gather(*tasks, return_exceptions=True)

    approved_data = []
    success_count = 0
    fail_count = 0

    for res in results:
        if isinstance(res, Exception):
            logger.critical(f"Worker crashed with error: {res}")
            continue

        if res.get('safety_check') == 'pass':
            clean_json = res['extraction'].model_dump()
            clean_json['filename'] = res['image_path']
            approved_data.append(clean_json)
            success_count += 1
        else:
            fail_count += 1

    output_file = "approved_invoices.json" if test_mode is None else "approved_invoices_donut.json"
    existing_data = []

    if os.path.exists(output_file):
        try:
            with open(output_file, "r") as f:
                file_content = json.load(f)
                if isinstance(file_content, list):
                    existing_data = file_content
        except (json.JSONDecodeError, OSError):
            pass  # Start fresh if file is corrupt or unreadable

    existing_data.extend(approved_data)

    with open(output_file, "w") as f:
        json.dump(existing_data, f, indent=2, default=str)

    logger.info(f"Complete.")
    logger.info(f"Success: {success_count} | Flagged for human review: {fail_count}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Invoice Processing CLI")
    parser.add_argument("--input_path", required=True, type=str, help="Path to an image file or directory of images")
    parser.add_argument("--num_agents", required=True, type=int, default=1, help="Number of agents to work in parallel")
    args = parser.parse_args()

    input_path = args.input_path
    num_agents = args.num_agents

    app = compile_workflow()
    asyncio.run(main(app, input_path, num_agents))
