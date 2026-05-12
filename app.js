document.addEventListener('DOMContentLoaded', () => {
    const calculateBtn = document.getElementById('calculate-btn');
    const resetBtn = document.getElementById('reset-btn');
    const resultsPanel = document.getElementById('results-panel');
    const inputPanel = document.querySelector('.input-panel');

    const BAD_INGREDIENTS = [
        'high fructose corn syrup', 'hydrogenated', 'aspartame', 'sucralose', 
        'saccharin', 'red 40', 'yellow 5', 'yellow 6', 'blue 1', 'bha', 'bht', 
        'sodium nitrate', 'sodium nitrite', 'potassium bromate', 'titanium dioxide'
    ];

    const UNHEALTHY_INGREDIENTS = [
        'added sugar', 'cane sugar', 'sugar', 'palm oil', 'enriched flour', 
        'refined wheat', 'maltodextrin', 'dextrose', 'soybean oil', 'canola oil', 
        'corn syrup', 'artificial flavor'
    ];

    const lookupBtn = document.getElementById('lookup-btn');
    const scanBtn = document.getElementById('scan-btn');
    const barcodeInput = document.getElementById('barcode-input');
    const barcodeStatus = document.getElementById('barcode-status');
    let html5QrCode;

    calculateBtn.addEventListener('click', calculateScore);
    resetBtn.addEventListener('click', resetApp);
    lookupBtn.addEventListener('click', () => fetchProduct(barcodeInput.value));
    scanBtn.addEventListener('click', toggleScanner);

    function calculateScore() {
        const calories = parseFloat(document.getElementById('calories').value);
        const protein = parseFloat(document.getElementById('protein').value);
        const carbs = parseFloat(document.getElementById('carbs').value);
        const fiber = parseFloat(document.getElementById('fiber').value) || 0;
        const fat = parseFloat(document.getElementById('fat').value);
        const ingredientsText = document.getElementById('ingredients').value.toLowerCase();

        if (isNaN(calories) || isNaN(protein) || isNaN(carbs) || isNaN(fat)) {
            alert("Please enter valid numbers for core nutrition facts.");
            return;
        }

        const netCarbs = Math.max(0, carbs - fiber);
        const calculatedCalories = (protein * 4) + (netCarbs * 4) + (fat * 9);
        const effectiveCalories = Math.max(calories, calculatedCalories, 1);
        
        let scoreLog = [];
        let baseScore = 5.0;
        scoreLog.push({ msg: 'Baseline Score', val: 5.0 });

        const proteinRatio = (protein * 4) / effectiveCalories;
        const carbRatio = (netCarbs * 4) / effectiveCalories;

        // Trivial Calories Bypass
        let bypassCarbs = false;
        if (calories < 30 && carbs < 5) {
            bypassCarbs = true;
            baseScore += 2.0;
            scoreLog.push({ msg: 'Very low calorie product', val: 2.0 });
        }

        // Single Ingredient / Healthy Fat Bonus (Graza EVOO)
        if (proteinRatio === 0 && carbRatio === 0 && fat > 0) {
            if (ingredientsText.includes('extra virgin olive oil') || ingredientsText.includes('avocado oil') || ingredientsText.includes('coconut oil')) {
                baseScore += 3.5;
                scoreLog.push({ msg: 'Single-ingredient healthy superfood fat', val: 3.5 });
            } else if (ingredientsText.includes('olive oil')) {
                baseScore += 2.0;
                scoreLog.push({ msg: 'Heart-healthy fat base', val: 2.0 });
            } else {
                baseScore += 1.0;
                scoreLog.push({ msg: 'Zero glycemic impact (all fat)', val: 1.0 });
            }
        }

        // Protein Scoring
        if (proteinRatio >= 0.6) {
            baseScore += 4.0;
            scoreLog.push({ msg: 'Exceptional lean protein density', val: 4.0 });
        } else if (proteinRatio >= 0.4) {
            baseScore += 3.5;
            scoreLog.push({ msg: 'Excellent protein-to-calorie ratio', val: 3.5 });
        } else if (proteinRatio >= 0.2) {
            baseScore += 2.0;
            scoreLog.push({ msg: 'Great protein source', val: 2.0 });
        } else if (proteinRatio >= 0.1) {
            baseScore += 1.0;
            scoreLog.push({ msg: 'Some protein', val: 1.0 });
        }

        // Fiber Scoring
        const fiberPer100Cal = (fiber / effectiveCalories) * 100;
        if (fiberPer100Cal >= 3) {
            baseScore += 2.0;
            scoreLog.push({ msg: 'Excellent fiber content', val: 2.0 });
        } else if (fiberPer100Cal >= 1.5) {
            baseScore += 1.0;
            scoreLog.push({ msg: 'Good fiber content', val: 1.0 });
        } else if (fiberPer100Cal < 0.5 && carbRatio > 0.5 && !bypassCarbs) {
            baseScore -= 0.5;
            scoreLog.push({ msg: 'High carbohydrates with almost no fiber', val: -0.5 });
        }

        // Natural Sweets / Fruits
        if (ingredientsText.includes('stevia') || ingredientsText.includes('erythritol') || ingredientsText.includes('monk fruit')) {
            baseScore += 1.5;
            scoreLog.push({ msg: 'Uses natural, non-glycemic sweetener', val: 1.5 });
        }
        if (ingredientsText.includes('orange') || ingredientsText.includes('apple') || ingredientsText.includes('spinach') || ingredientsText.includes('kale') || ingredientsText.includes('fruit')) {
            baseScore += 2.0;
            scoreLog.push({ msg: 'Natural whole food/fruit base', val: 2.0 });
        }
        if (ingredientsText.includes('cheese')) {
            baseScore += 0.5;
            scoreLog.push({ msg: 'Contains real cheese/dairy', val: 0.5 });
        }

        // Carb & Sugar Penalties
        if (!bypassCarbs) {
            if (carbRatio > 0.7) {
                baseScore -= 1.5;
                scoreLog.push({ msg: 'Extremely high proportion of carbohydrates', val: -1.5 });
            } else if (carbRatio > 0.5) {
                baseScore -= 1.0;
                scoreLog.push({ msg: 'High carbohydrate ratio', val: -1.0 });
            }

            const hasSugar = ingredientsText.includes("sugar") || ingredientsText.includes("syrup") || ingredientsText.includes("dextrose") || ingredientsText.includes("maltodextrin");
            if (carbRatio > 0.4 && hasSugar) {
                baseScore -= 1.5;
                scoreLog.push({ msg: 'High carbs mostly from refined sugars', val: -1.5 });
            }
        }

        // Ingredients Scanning
        let badPenalty = 0;
        let unhealthyPenalty = 0;
        const foundBad = [];
        const foundUnhealthy = [];

        BAD_INGREDIENTS.forEach(ingredient => {
            if (ingredientsText.includes(ingredient)) {
                foundBad.push(ingredient);
                badPenalty += 1.5;
                scoreLog.push({ msg: `Hazardous ingredient: ${ingredient}`, val: -1.5 });
            }
        });

        UNHEALTHY_INGREDIENTS.forEach(ingredient => {
            const isSubStringOfBad = foundBad.some(bad => bad.includes(ingredient));
            if (!isSubStringOfBad && ingredientsText.includes(ingredient)) {
                foundUnhealthy.push(ingredient);
                unhealthyPenalty += 1.0;
                scoreLog.push({ msg: `Unhealthy ingredient: ${ingredient}`, val: -1.0 });
            }
        });

        baseScore -= (badPenalty + unhealthyPenalty);
        
        let finalScore = Math.max(1.0, Math.min(10.0, baseScore));
        finalScore = Math.round(finalScore * 10) / 10; // Round to 1 decimal place

        // Update UI
        updateUI(finalScore, scoreLog);
    }

    function updateUI(score, scoreLog) {
        inputPanel.style.display = 'none';
        resultsPanel.style.display = 'block';
        resultsPanel.classList.add('animate-slide-in');

        const scoreElement = document.getElementById('final-score');
        const scoreCircle = document.querySelector('.score-circle');
        const messageElement = document.getElementById('score-message');

        // Animate counter
        let currentCount = 0;
        const duration = 1000;
        const interval = 20;
        const steps = duration / interval;
        const increment = score / steps;

        const counter = setInterval(() => {
            currentCount += increment;
            if (currentCount >= score) {
                currentCount = score;
                clearInterval(counter);
            }
            scoreElement.innerText = currentCount.toFixed(1);
        }, interval);

        let mainColor = '';
        if (score >= 8.0) {
            mainColor = 'var(--accent-color)';
            messageElement.innerText = 'Great Choice! Very Healthy.';
            messageElement.style.color = mainColor;
        } else if (score >= 5.0) {
            mainColor = 'var(--warning-color)';
            messageElement.innerText = 'Moderate. Consume in moderation.';
            messageElement.style.color = mainColor;
        } else {
            mainColor = 'var(--danger-color)';
            messageElement.innerText = 'Poor. Avoid if possible.';
            messageElement.style.color = mainColor;
        }

        const percentage = (score / 10.0) * 100;
        setTimeout(() => {
            scoreCircle.style.background = `conic-gradient(${mainColor} ${percentage}%, rgba(255,255,255,0.05) 0%)`;
            scoreCircle.style.boxShadow = `inset 0 0 20px rgba(0,0,0,0.5), 0 0 30px ${mainColor}40`;
        }, 100);

        // Update Macro Bar
        const macroBar = document.getElementById('macro-bar');
        macroBar.style.width = `${percentage}%`;
        macroBar.style.backgroundColor = mainColor;

        // Render Breakdown List
        const breakdownList = document.getElementById('breakdown-list');
        breakdownList.innerHTML = '';
        
        scoreLog.forEach(log => {
            const li = document.createElement('li');
            const cls = log.val > 0 ? 'positive' : (log.val < 0 ? 'negative' : 'neutral');
            li.className = cls;
            
            const msgSpan = document.createElement('span');
            msgSpan.innerText = log.msg;
            
            const valSpan = document.createElement('span');
            valSpan.style.fontWeight = 'bold';
            valSpan.innerText = log.val > 0 ? `+${log.val}` : `${log.val}`;
            
            li.appendChild(msgSpan);
            li.appendChild(valSpan);
            breakdownList.appendChild(li);
        });
    }

    function resetApp() {
        resultsPanel.style.display = 'none';
        inputPanel.style.display = 'block';
        const scoreCircle = document.querySelector('.score-circle');
        scoreCircle.style.background = `conic-gradient(var(--accent-color) 0%, transparent 0%)`;
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    async function fetchProduct(barcode) {
        if (!barcode) {
            barcodeStatus.innerText = "Please enter a barcode.";
            return;
        }
        barcodeStatus.innerText = "Fetching...";
        barcodeStatus.style.color = "var(--text-muted)";
        
        try {
            const res = await fetch(`https://world.openfoodfacts.org/api/v0/product/${barcode}.json`);
            const data = await res.json();
            
            if (data.status === 1) {
                const p = data.product;
                const nut = p.nutriments || {};
                
                document.getElementById('calories').value = nut['energy-kcal_100g'] || nut['energy-kcal_serving'] || 0;
                document.getElementById('protein').value = nut['proteins_100g'] || nut['proteins_serving'] || 0;
                document.getElementById('carbs').value = nut['carbohydrates_100g'] || nut['carbohydrates_serving'] || 0;
                document.getElementById('fiber').value = nut['fiber_100g'] || nut['fiber_serving'] || 0;
                document.getElementById('fat').value = nut['fat_100g'] || nut['fat_serving'] || 0;
                
                document.getElementById('ingredients').value = p.ingredients_text_en || p.ingredients_text || "";
                
                barcodeStatus.innerText = `Found: ${p.product_name || 'Unknown Product'}`;
                barcodeStatus.style.color = "var(--accent-color)";
                
                calculateScore();
            } else {
                barcodeStatus.innerText = "Product not found. Please enter manually.";
                barcodeStatus.style.color = "var(--warning-color)";
            }
        } catch (err) {
            barcodeStatus.innerText = "Error fetching product.";
            barcodeStatus.style.color = "var(--danger-color)";
        }
    }

    function toggleScanner() {
        const readerDiv = document.getElementById('reader');
        if (readerDiv.style.display === 'block') {
            readerDiv.style.display = 'none';
            if (html5QrCode) {
                html5QrCode.stop().catch(err => console.log(err));
            }
        } else {
            readerDiv.style.display = 'block';
            if (!html5QrCode) {
                html5QrCode = new Html5Qrcode("reader");
            }
            html5QrCode.start(
                { facingMode: "environment" },
                { fps: 10, qrbox: { width: 250, height: 150 } },
                (decodedText, decodedResult) => {
                    document.getElementById('barcode-input').value = decodedText;
                    toggleScanner(); 
                    fetchProduct(decodedText);
                },
                (errorMessage) => {
                }
            ).catch((err) => {
                barcodeStatus.innerText = "Camera access denied or error.";
                barcodeStatus.style.color = "var(--danger-color)";
            });
        }
    }
});
