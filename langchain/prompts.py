class Prompts:
    @staticmethod
    def get_graph_prompt(dataset_name, relationships_list):
        """Return prompt string to create knowledge graph."""
        prompt = None
        if dataset_name == 'clutrr':
            prompt = (
                "You are an expert in relationship triple construction regarding"
                " families and relationships as (person_A, relationship, person_B).\n"
                "You have to create relationship triples extracting all facts from the following statement,\n\n"
                "{statement}\n\n"
                "The relationships in the graph has to be in this list : \n\n "
                + str(relationships_list) +
                "\n\n Infer the inverse relationships and add them to the response as well. \n\n"
            )
        return prompt

    @staticmethod
    def get_relationship_prompt(dataset_name, entity_classes_list):
        prompt = None
        if dataset_name == 'clutrr':
            prompt = [
                ("system", f"""
            <system>
              <role>
                You are an expert in understanding family and social relationships.
                Your goal is to analyze a chain of known relationships and deduce the
                single direct relationship that links the first person to the last person.
              </role>

              <behavior>
                <rule name=\"Valid-terms-only\">
                  The predefined list of valid relationship types (case-insensitive) is
                  authoritative. Every intermediate and final label you produce MUST come
                  from that list. Never invent synonyms or new terms.
                </rule>

                <rule name=\"Sequential-traversal\">
                  Read the input sequence in order. After each hop, update the currently
                  known relationship between {{first_person}} and the interim target.
                  Record this evolution openly in your reasoning.
                </rule>

                <rule name=\"No-social-heuristics\">
                  Do NOT apply cultural shortcuts (e.g., “sibling of a grandson is niece”)
                  or infer roles absent from the explicit chain. Derive every step purely
                  from the provided relationships and valid terms.
                </rule>

                <rule name=\"Single-word-answer\">
                  Your final answer must be exactly one term—spelled as it appears in the
                  valid list—that best describes how {{last_person}} relates to
                  {{first_person}}. Do not add qualifiers or punctuation.
                </rule>
              </behavior>

              <format>
                1. Produce a detailed, step-by-step reasoning section that:
                   • cites each hop in the input chain,
                   • states the intermediate relationship using ONLY valid terms,
                   • shows how the relationship evolves until {{last_person}}.
                2. Then print this heading on its own line (exactly as written):
                   ### FINAL ANSWER
                3. On the very next line, output ONLY the single-word answer from the
                   valid list—no quotes, no extra text.
                4. End without adding anything after that word.
              </format>
            </system>
            """),
                ("human", f"""
            Valid relationship terms (case-insensitive):
            {entity_classes_list}

            Input sequence:
            {{relation_str}}

            → What is the relationship from **{{first_person}}** to **{{last_person}}**?
            (Remember to follow the output format.)""")
            ]
        return prompt

    @staticmethod
    def get_revision_prompt(dataset_name, entity_classes_list):
        prompt = None
        if dataset_name == 'clutrr':
            prompt = f"""
            You are an expert in understanding family relationships. You are given:

            1. A list of relationships between individuals in the form of (PersonA, Relationship, PersonB).
            2. A selected answer for the relationship ({{first_person}}, ?, {{last_person}}).
            3. Reasoning steps to derive the answer.

            Your task is to check the correctness of the provided answer and the previous reasoning steps and, if necessary, revise it. You will also provide a step-by-step reasoning for your revised answer.

            Follow these steps:

            1. Go through the predefined list of valid relationship types and grasp the meaning of each term. Consider this list as a case-insensitive list of relationship terms.
            2. Traverse the list of relationships and the previous reasoning steps step-by-step. At each step, try to identify logical errors in the defined relationships. Correct them if necessary in the revised answer and reasoning.
            3. When giving the revised answer and reasoning, use only terms from the valid list to describe each intermediate and final relationship. **Do not infer social roles or make assumptions that are not strictly derived from the relationship sequence.**
            4. Provide your final answer in one word, **strictly** from the valid list, and ensure it is logically consistent with your reasoning steps.

            Valid answers:
            {entity_classes_list}

            Input sequence:
            {{relation_str}}

            Selected answer:
            {{selected_answer}}

            Reasoning steps:
            {{reasoning_steps}}

            Return your response as a JSON object in the following format:

            ```json keys :-
               \"previous_reasoning_errors\": \"<list of errors found in the previous reasoning steps>\",
               \"reason\": \"<step-by-step reasoning for the revised answer>\",
               \"revised_answer\": \"<single-word answer from valid list>\"
            ```
            """
        return prompt

    @staticmethod
    def get_extraction_prompt(relationships_list):
        return [
            ("system", f"""
        <system>
          <role>
            You are an expert in natural language understanding and human relationship extraction.
            Your purpose is to read a user-provided statement and output clean, name-only triples.
          </role>

          <behavior>
            <rule name=\"Spouse-detection\">
              Whenever you see “[Name] fixed her husband [OtherName]” or “[Name] cooked his wife [OtherName] …”,
              infer a spousal relationship:
              (Name, has_husband, OtherName) and (OtherName, has_wife, Name) if “her husband”,
              or (Name, has_wife, OtherName) and (OtherName, has_husband, Name) if “his wife.”
            </rule>

            <rule name=\"Mother’s-husband\">
              Whenever you see a phrase of the form “&lt;bracketed Name&gt;’s mother’s husband [OtherName]”,
              infer that OtherName is that bracketed Name’s father:
              (Name, has_father, OtherName) and (OtherName, has_son, Name).
            </rule>

            <rule name=\"Father’s\">
              Whenever you see a phrase of the form “&lt;bracketed Name&gt;’s father, [OtherName]”,
              infer that OtherName is that bracketed Name’s father:
              (Name, has_father, OtherName) and (OtherName, has_daughter, Name).
            </rule>

            <rule name=\"Son-naming\">
              Whenever you see “&lt;bracketed Name&gt;’s son, [OtherName]” (for example, “John’s son, [Bryan]”),
              produce (OtherName, has_father, Name) and (Name, has_son, OtherName).
            </rule>

            <rule name=\"Dad/Father-naming\">
              Whenever you see “&lt;bracketed Name&gt;’s dad, [OtherName]” (or “&lt;bracketed Name&gt;’s father, [OtherName]”),
              produce (Name, has_father, OtherName) and (OtherName, has_daughter, Name).
            </rule>

            <rule name=\"Naming-a-daughter\">
              Whenever you see “&lt;bracketed Parent&gt; had a daughter named [Child]”,
              infer (Parent, has_daughter, Child) and (Child, has_mother, Parent).
            </rule>

            <rule name=\"Sister-sibling\">
              Whenever you see “[Name1] and his sister [Name2] …” or “[Name1] and her sister [Name2] …”,
              infer (Name1, has_sister, Name2) and (Name2, has_brother, Name1) if Name1 is male,
              or (Name1, has_sister, Name2) and (Name2, has_sister, Name1) if both are female.
              Otherwise, do not infer a sibling link.
            </rule>

            <rule name=\"Uncle\">
              Whenever you see “&lt;bracketed NieceOrNephew&gt; went to her uncle [OtherName]’s house” (or any similar phrasing),
              infer (NieceOrNephew, has_uncle, OtherName) and (OtherName, has_niece, NieceOrNephew).
            </rule>

            <rule name=\"Shared-Parent Siblings\">
              Whenever you see two bracketed names appear together and share a parent phrase:
                • “[A] and [B] asked their mother [C] …”
                • “[A] and [B] went to see their father [C] …”
              then:
                1. If [C] is explicitly named in brackets, infer each child’s parent link:
                   (A, has_mother, C) / (B, has_mother, C) if it’s “their mother [C]”,
                   or (A, has_father, C) / (B, has_father, C) if it’s “their father [C]”.
                2. Because A and B share the same named parent [C], infer they are siblings:
                   – If A is male and B is female: (A, has_sister, B) and (B, has_brother, A).
                   – If both are male: (A, has_brother, B) and (B, has_brother, A).
                   – If both are female: (A, has_sister, B) and (B, has_sister, A).
                   – If one’s gender cannot be determined, do not infer a sibling triple.
              (If the shared parent [C] is not named in brackets, do not create the parent link—only infer siblings if there’s a clear pronoun “their mother/father.”)
            </rule>

            <rule name=\"Grandson-naming\">
              Whenever you see “&lt;bracketed Name&gt; has … grandson. The grandson’s name is [OtherName]”,
              infer (Name, has_grandson, OtherName) and (OtherName, has_grandfather, Name).
            </rule>

            <rule name=\"Other-explicit-relationships\">
              Identify all other explicitly stated or strongly implied family relationships—this includes:
              • Direct parent/child (has_mother / has_father / has_son / has_daughter) only if none of the above rules applies.
              • Sibling (has_brother / has_sister) only if the text literally says “brother” or “sister,” or via the sibling rules above.
              • Aunt/uncle/niece/nephew.
              • Grandparent/grandchild (has_grandmother / has_grandfather / has_grandson / has_granddaughter) only if not already covered by “Grandson-naming.”
            </rule>

            <rule name=\"No-invent\">
              Do not invent any new names or assume any relationships beyond what is explicitly stated or clearly implied.
            </rule>
          </behavior>

          <format>
            1. First, provide a detailed, step-by-step reasoning showing how you:
               • Resolved each special phrase (e.g., “X’s father, [Y]”, “had a daughter named [Y]”, “A and his sister, [B]”, “has … grandson. The grandson’s name is [Y]”, etc.).
               • Determined gender when pronouns like “her” or “his” appeared (or used name-based inference if no pronoun).
               • Decided which rule applied (Spouse-detection, Mother’s-husband, Father’s, Son-naming, Dad/Father-naming, Naming-a-daughter, Sister-sibling, Uncle, Shared-Parent Siblings, Grandson-naming, Other-explicit-relationships, etc.).
               • Extracted each family link and mapped it to exactly one relation from the allowed list.

            2. Then, print exactly this heading on its own line (no extra text):
               ### TRIPLES_START

            3. Below that line, list every relationship as exactly two triples—one direct and one inverse—using only the bracketed names and parentheses. Do NOT use any quotes or square brackets. Each triple must follow this format:
               (Subject, relationship, Object)
               And “relationship” must come from this list (exactly as written, with underscores):
               {relationships_list}

            4. Finally, after listing all triples, end without adding any extra text or parentheses.
          </format>
        </system>
        """),
            ("human", "{statement}"),
        ]
