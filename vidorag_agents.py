from PIL import Image
import os
import json
import sys

from agent.agent_prompt import seeker_prompt,inspector_prompt,answer_prompt
from agent.map_dict import arrangement_map_dict,page_map_dict_normal,page_map_dict

from utils.parse_tool import extract_json
from utils.image_preprosser import concat_images_with_bbox

def _trace_add(trace, agent, event, **kwargs):
    if trace is None:
        return
    item = {
        "agent": agent,
        "event": event,
    }
    item.update(kwargs)
    trace.append(item)

def _truncate_text(text, max_len=1500):
    if text is None:
        return None
    text = str(text)
    if len(text) <= max_len:
        return text
    return text[:max_len] + "...<truncated>"

class Seeker:
    def __init__(self, vlm):
        self.vlm = vlm
        self.seeker_multi_image = True

        if self.seeker_multi_image:
            self.page_map = page_map_dict_normal
        else:
            self.page_map = page_map_dict
    def run(self, query=None, images_path=None, feedback=None, trace=None):

        if query is not None and images_path is not None:
            self.buffer_images = images_path
            self.query = query
            prompt = seeker_prompt.replace('{question}', self.query).replace('{page_map}', self.page_map[len(self.buffer_images)])
            _trace_add(
                trace,
                agent="seeker",
                event="start",
                mode="initial",
                candidate_images=list(self.buffer_images),
                prompt=_truncate_text(prompt),
            )
        
        elif feedback is not None:
            additional_information = self.query + '\n\n## Additional Information\n' + feedback
            prompt = seeker_prompt.replace('{question}', additional_information).replace('{page_map}', self.page_map[len(self.buffer_images)])
            _trace_add(
                trace,
                agent="seeker",
                event="start",
                mode="feedback",
                feedback=feedback,
                candidate_images=list(self.buffer_images),
                prompt=_truncate_text(prompt),
            )

        if self.seeker_multi_image:
            input_images = self.buffer_images
        else:
            input_images = [concat_images_with_bbox(self.buffer_images, arrangement=arrangement_map_dict[len(self.buffer_images)], scale=1, line_width=40)]

        times = 0
        while True:
            if times > 2:
                # return None, None, None
                raise Exception('seeker time out')
            times += 1
            select_response = self.vlm.generate(query=prompt, image=input_images, mode="seeker")
            print(select_response)
            _trace_add(
                trace,
                agent="seeker",
                event="llm_response",
                attempt=times,
                raw_response=_truncate_text(select_response),
            )
            try:
                select_response_json = extract_json(select_response)
                reason = select_response_json.get('reason', None)
                summary = select_response_json.get('summary', None)
                select_page_num = select_response_json.get('choice', None)
                if reason is None or summary is None or select_page_num is None:
                    raise Exception('select json format error: missing fields')

                valid_page_num = _normalize_indices(
                    select_page_num,
                    len(self.buffer_images),
                    label="indices",
                    empty_fallback="all",
                )
                selected_images = [self.buffer_images[page] for page in valid_page_num]
                self.buffer_images = [image for image in self.buffer_images if image not in selected_images]
                _trace_add(
                    trace,
                    agent="seeker",
                    event="selection",
                    attempt=times,
                    reason=reason,
                    summary=summary,
                    choice=select_page_num,
                    normalized_choice=valid_page_num,
                    selected_images=selected_images,
                    remaining_images=list(self.buffer_images),
                )

            except Exception as e:
                print(e)
                print(select_response)
                print('seeker')
                _trace_add(
                    trace,
                    agent="seeker",
                    event="parse_error",
                    attempt=times,
                    error=str(e),
                )
                continue
            break
        
        return selected_images, summary, reason

class Inspector:
    def __init__(self, vlm):
        self.vlm = vlm
        self.inspector_multi_image = True

        if self.inspector_multi_image:
            self.page_map = page_map_dict_normal
        else:
            self.page_map = page_map_dict

        self.buffer_images = []

    def run(self, query, images_path, trace=None):
        # answer or not, (images, candidate)/feedback
        if len(self.buffer_images) == 0 and len(images_path) == 0:
            _trace_add(trace, agent="inspector", event="empty_input")
            return None, None, None
        elif len(images_path) == 0:
            _trace_add(
                trace,
                agent="inspector",
                event="route",
                status="synthesizer",
                reason="no_new_images_use_buffer",
                buffer_images=list(self.buffer_images),
            )
            return 'synthesizer', None, self.buffer_images
        elif len(images_path) != 0:
            self.buffer_images.extend(images_path)
            _trace_add(
                trace,
                agent="inspector",
                event="start",
                input_images=list(images_path),
                merged_buffer=list(self.buffer_images),
            )

        if self.inspector_multi_image:
            input_images = self.buffer_images
        else:
            input_images = [concat_images_with_bbox(self.buffer_images, arrangement=arrangement_map_dict[len(self.buffer_images)], scale=1, line_width=40)]

        prompt = inspector_prompt.replace('{question}',query).replace('{page_map}',self.page_map[len(self.buffer_images)])

        times = 0
        while True:
            if times >2:
                raise Exception('Inspector time out')
                # return None, None, None
            times +=1

            response = self.vlm.generate(query=prompt,image=input_images, mode="inspector")
            print(response)
            _trace_add(
                trace,
                agent="inspector",
                event="llm_response",
                attempt=times,
                raw_response=_truncate_text(response),
            )
            try:
                response_json = extract_json(response)
                # thought
                reason = response_json.get('reason',None)
                # if feedback
                info = response_json.get('information',None)
                choice = response_json.get('choice',None)
                # can answer
                answer = response_json.get('answer',None)
                ref = response_json.get('reference',None)

                if reason is None:
                    raise Exception('answer no reason')
                elif answer is not None and ref is not None:
                    valid_ref = _normalize_indices(
                        ref,
                        len(self.buffer_images),
                        label="reference",
                        empty_fallback="error",
                    )
                    if len(valid_ref) == len(self.buffer_images):
                        _trace_add(
                            trace,
                            agent="inspector",
                            event="route",
                            status="answer",
                            attempt=times,
                            reason=reason,
                            answer=answer,
                            reference=ref,
                            normalized_reference=valid_ref,
                        )
                        return 'answer', answer, self.buffer_images
                    else:
                        ref_images = [self.buffer_images[page] for page in valid_ref]
                        _trace_add(
                            trace,
                            agent="inspector",
                            event="route",
                            status="synthesizer",
                            attempt=times,
                            reason=reason,
                            answer=answer,
                            reference=ref,
                            normalized_reference=valid_ref,
                            selected_images=ref_images,
                        )
                        return 'synthesizer', answer, ref_images
                elif info is not None and choice is not None:
                    if len(choice) == 0:
                        _trace_add(
                            trace,
                            agent="inspector",
                            event="route",
                            status="seeker",
                            attempt=times,
                            reason=reason,
                            information=info,
                            choice=choice,
                            selected_images=list(self.buffer_images),
                        )
                        return 'seeker', info, self.buffer_images
                    valid_choice = _normalize_indices(
                        choice,
                        len(self.buffer_images),
                        label="choice",
                        empty_fallback="error",
                    )
                    self.buffer_images = [self.buffer_images[page] for page in valid_choice]
                    _trace_add(
                        trace,
                        agent="inspector",
                        event="route",
                        status="seeker",
                        attempt=times,
                        reason=reason,
                        information=info,
                        choice=choice,
                        normalized_choice=valid_choice,
                        selected_images=list(self.buffer_images),
                    )
                    return 'seeker', info, self.buffer_images
            
            except Exception as e:
                print(e)
                print("inspector")
                _trace_add(
                    trace,
                    agent="inspector",
                    event="parse_error",
                    attempt=times,
                    error=str(e),
                )
                continue

class Synthesizer:
    def __init__(self, vlm):
        self.vlm = vlm
        self.synthesizer_multi_image = True
        if self.synthesizer_multi_image:
            self.page_map = page_map_dict_normal
        else:
            self.page_map = page_map_dict

    def run(self, query, candidate_answer, ref_images, trace=None):
        if candidate_answer is not None:
            query = query + '\n\n## Related Information\n' + candidate_answer
        prompt = answer_prompt.replace('{question}',query).replace('{page_map}',self.page_map[len(ref_images)])
        _trace_add(
            trace,
            agent="synthesizer",
            event="start",
            candidate_answer=candidate_answer,
            ref_images=list(ref_images),
            prompt=_truncate_text(prompt),
        )

        if self.synthesizer_multi_image:
            input_images = ref_images
        else:
            input_images = [concat_images_with_bbox(ref_images, arrangement=arrangement_map_dict[len(ref_images)], scale=1, line_width=40)]
        
        times = 0
        while True:
            times += 1
            final_answer_response = self.vlm.generate(query=prompt,image=input_images, mode="synthesizer")
            print(final_answer_response)
            _trace_add(
                trace,
                agent="synthesizer",
                event="llm_response",
                attempt=times,
                raw_response=_truncate_text(final_answer_response),
            )
            try:
                final_answer_response_json = extract_json(final_answer_response)
                reason = final_answer_response_json.get('reason',None)
                answer = final_answer_response_json.get('answer',None)
                if reason is None or answer is None:
                    raise Exception('answer missing')
                _trace_add(
                    trace,
                    agent="synthesizer",
                    event="final_answer",
                    attempt=times,
                    reason=reason,
                    answer=answer,
                )
                return reason, answer
            except Exception as e:
                print(e)
                print(final_answer_response)
                print("answer")
                _trace_add(
                    trace,
                    agent="synthesizer",
                    event="parse_error",
                    attempt=times,
                    error=str(e),
                )
                continue

class ViDoRAG_Agents:
    def __init__(self, vlm):
        self.seeker = Seeker(vlm)
        self.inspector = Inspector(vlm)
        self.synthesizer = Synthesizer(vlm)

    def run_agent(self, query, images_path, return_trace=False):
        # initial
        self.seeker.buffer_images = None
        self.inspector.buffer_images = []
        trace = [] if return_trace else None

        selected_images, summary, reason = self.seeker.run(query=query, images_path=images_path, trace=trace)
        # iter
        step = 0
        while True:
            step += 1
            _trace_add(
                trace,
                agent="coordinator",
                event="loop",
                step=step,
                selected_images=list(selected_images) if selected_images is not None else None,
            )
            status, information, images = self.inspector.run(query, selected_images, trace=trace)
            if status == 'answer':
                if return_trace:
                    _trace_add(trace, agent="coordinator", event="finish", status="answer", answer=information)
                    return information, trace
                return information
            elif status == 'synthesizer':
                reason, answer = self.synthesizer.run(query, information, images, trace=trace)
                if return_trace:
                    _trace_add(trace, agent="coordinator", event="finish", status="synthesizer", answer=answer, reason=reason)
                    return answer, trace
                return answer
            elif status == 'seeker':
                selected_images, summary, reason = self.seeker.run(feedback=information, trace=trace)
                continue
            else:
                print('No related information')
                if return_trace:
                    _trace_add(trace, agent="coordinator", event="finish", status="none", answer=None)
                    return None, trace
        if return_trace:
            return None, trace
        return None

def _normalize_indices(indices, buffer_len, label="indices", empty_fallback="error"):
    valid_indices = [page for page in indices if page < buffer_len]
    if len(valid_indices) < len(indices):
        # Heuristic: model may be using 1-based indices.
        if all(isinstance(p, int) for p in indices) and min(indices) >= 1 and max(indices) <= buffer_len:
            valid_indices = [p - 1 for p in indices]
            print(f"Warning: Converted 1-based {label} {indices} to 0-based for length {buffer_len}")
        else:
            print(f"Warning: Filtered out invalid {label} {indices} for buffer length {buffer_len}")
    if len(valid_indices) == 0:
        if empty_fallback == "all":
            print(f"Warning: Empty {label}; falling back to all {buffer_len} images")
            valid_indices = list(range(buffer_len))
        else:
            raise Exception(f'{label} error')

    # de-duplicate while preserving order
    seen = set()
    valid_indices = [p for p in valid_indices if not (p in seen or seen.add(p))]
    return valid_indices

if __name__ == '__main__':
    from llms.llm import LLM
    vlm = LLM('openbmb/EVisRAG-7B')
    agent = ViDoRAG_Agents(vlm)
    re=agent.run_agent(query='Who is Tim?', images_path=['./data/ExampleDataset/img/00a76e3a9a36255616e2dc14a6eb5dde598b321f_1.jpg','./data/ExampleDataset/img/00a76e3a9a36255616e2dc14a6eb5dde598b321f_2.jpg','./data/ExampleDataset/img/00a76e3a9a36255616e2dc14a6eb5dde598b321f_3.jpg','./data/ExampleDataset/img/00a76e3a9a36255616e2dc14a6eb5dde598b321f_4.jpg'])
    print(re)
