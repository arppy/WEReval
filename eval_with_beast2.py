from playwright.sync_api import sync_playwright

audio_files = ["/home/berta/data/HungarianDysartriaDatabase/wav/C_001_0001_computer_NULL_AO_2.wav"]

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto("https://phon.nytud.hu/beast2")
    for audio_file in audio_files:
        # 1. Clear if needed (optional)
        page.click("#component-4")

        # 2. Upload
        page.set_input_files("#component-2 input[type='file']", audio_file)

        # 3. Wait for waveform/duration → upload complete
        page.wait_for_selector("#component-2 .waveform-container", timeout=30000)

        # 4. Now click Run — safe!
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
        print(audio_file,result.strip())
    browser.close()