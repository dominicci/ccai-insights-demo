
import pandas as pd
import sys

def parse_transcript(transcript):
    """Parses transcript to extract key indicators."""
    transcript_lower = str(transcript).lower()
    
    # 1. Determine Issue Type
    issue_type = "Other"
    if "preference" in transcript_lower or "wrong size" in transcript_lower or "color" in transcript_lower or "fit" in transcript_lower or "changed my mind" in transcript_lower:
        issue_type = "Preference"
    elif "defective" in transcript_lower or "broken" in transcript_lower or "not working" in transcript_lower or "won't sync" in transcript_lower or "blisters" in transcript_lower:
        if "blisters" in transcript_lower:
             issue_type = "Preference" 
        else:
             issue_type = "Technical"
    
    # 2. Determine Frustration
    frustrated = False
    frustration_keywords = ["ridiculous", "ugh", "finally", "angry", "disappointed", "terrible", "unacceptable", "blasting you", "manager"]
    if any(keyword in transcript_lower for keyword in frustration_keywords):
        frustrated = True
        
    # 3. Determine Empathy
    empathy = False
    empathy_keywords = ["i understand", "i'm sorry", "apologize", "completely understand", "hear that"]
    if any(keyword in transcript_lower for keyword in empathy_keywords):
        empathy = True
        
    # 4. Determine Discovery / Troubleshooting
    # Broaden logic to include "Discovery" questions as per refined Definition
    discovery_score = False
    discovery_keywords = [
        "what specifically is the issue", 
        "technical fault or a preference", 
        "troubleshoot",
        "have you tried",
        "reset",
        "reason for the return"
    ]
    if any(phrase in transcript_lower for phrase in discovery_keywords):
        discovery_score = True

    return {
        "issue_type": issue_type,
        "frustrated": frustrated,
        "empathy": empathy,
        "discovery_score": discovery_score
    }

def grade_row(row):
    transcript = row['Transcript']
    question = row['QaQuestionBody']
    
    context = parse_transcript(transcript)
    
    # Logic for: "Did the agent ask clarifying questions to understand the specific reason for the return or issue?"
    # (Matches "Mandatory Diagnosis / Discovery")
    if "clarifying questions" in question.lower():
        # Instruction: "Score N/A, if the customer voluntarily provides the detailed reason in their opening statement."
        # This is hard to detect perfectly with regex, but we can assume if the agent ASKS, it's a YES.
        # If the agent DOESN'T ask, it might be NO or N/A. 
        # For this script, strict grading: If they asked -> Yes. Else -> No (simple).
        # We can refine N/A logic if possible, but simpler is safer for bulk.
        if context["discovery_score"]:
            return "Yes"
        else:
            return "No"

    # Logic for: "Did the agent acknowledge the customer's frustration with empathy?"
    if "empathy" in question.lower():
        if not context["frustrated"]:
            return "N/A"
        if context["empathy"]:
            return "Yes"
        return "No"

    return "" # Unknown question

def main():
    files = [
        "data/upload_batches/call-center-transcripts-dataset_velofit_quality-scorecard-templates_callibration_test2_0.csv",
        "data/upload_batches/call-center-transcripts-dataset_velofit_quality-scorecard-templates_callibration_test2_1.csv"
    ]
    
    dfs = []
    print("Reading files...")
    for f in files:
        try:
            df_temp = pd.read_csv(f)
            dfs.append(df_temp)
            print(f"Loaded {len(df_temp)} rows from {f}")
        except Exception as e:
            print(f"Error reading {f}: {e}")
            return

    if not dfs:
        print("No data loaded.")
        return

    # Merge
    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"Total merged rows: {len(merged_df)}")
    
    # Apply grading
    print("Grading...")
    merged_df['QaAnswerValue'] = merged_df.apply(grade_row, axis=1)
    
    output_file = "data/upload_batches/call-center-transcripts-dataset_velofit_quality-scorecard-templates_callibration_merged.csv"
    merged_df.to_csv(output_file, index=False)
    print(f"Successfully saved graded merged file to {output_file}")
    
    # Verification
    print("\nSample Verification:")
    print(merged_df[['QaQuestionBody', 'QaAnswerValue']].sample(10))

if __name__ == "__main__":
    main()
