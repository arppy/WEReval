from playwright.sync_api import sync_playwright

audio_file = "/home/berta/data/HungarianDysartriaDatabase/wav/C_001_0001_computer_NULL_AO_2.wav"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto("https://phon.nytud.hu/beast2")

    # Upload audio directly
    page.set_input_files("#component-2 input[type='file']", audio_file)

    # Click Run
    page.click("#component-5")

    # Wait until the textarea has non-empty value
    page.wait_for_function("""
        () => {
            const textarea = document.querySelector('#component-10 textarea');
            return textarea && textarea.value.trim() !== '' && !textarea.value.match(/^\\d+\\.\\d+s$/);
        }
    """, timeout=60000)

    # Extract the actual value from the textarea
    result = page.eval_on_selector("#component-10 textarea", "el => el.value")
    print("Transcription:", result.strip())

    browser.close()