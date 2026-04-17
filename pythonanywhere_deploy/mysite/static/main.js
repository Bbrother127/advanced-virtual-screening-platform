// Global variables
var modelResultsData = null;
var modelColors = {};
var screeningResultsData = null;
var bestModelName = null;

var MODEL_COLORS = {
    'SVM': '#1f77b4',
    'LR': '#ff7f0e',
    'KNN': '#2ca02c',
    'RF': '#d62728',
    'PLS-DA': '#9467bd',
    'DFNN': '#8c564b'
};

var MODEL_KEY = {
    'SVM': 'svm', 'LR': 'lr', 'KNN': 'knn', 'RF': 'rf', 'PLS-DA': 'plsda', 'DFNN': 'dfnn'
};

var MODEL_NAMES = ['SVM', 'LR', 'KNN', 'RF', 'PLS-DA', 'DFNN'];

function scrollToSection(sectionId) {
    var el = document.getElementById(sectionId);
    if (el) { el.scrollIntoView({ behavior: 'smooth' }); }
}

function showStatus(elementId, message, type) {
    var el = document.getElementById(elementId);
    if (el) { el.className = 'status ' + type; el.innerHTML = message; }
}

function formatNum(num, dec) {
    if (dec === undefined) dec = 3;
    if (num === null || num === undefined || isNaN(num)) return '0.000';
    return num.toFixed(dec);
}

function getColor(name) {
    return MODEL_COLORS[name] || modelColors[name] || '#999';
}

function uploadTrainingData() {
    var fileInput = document.getElementById('training-file');
    var file = fileInput.files[0];
    if (!file) { showStatus('training-status', '\u8bf7\u9009\u62e9\u8bad\u7ec3\u6570\u636e\u6587\u4ef6', 'error'); return; }
    showStatus('training-status', '\u23f3 \u6b63\u5728\u4e0a\u4f20\u5e76\u8bad\u7ec3\u6a21\u578b\uff0c\u8bf7\u7a0d\u5019...', 'loading');
    var formData = new FormData();
    formData.append('file', file);
    fetch('/api/upload-training-data', { method: 'POST', body: formData })
    .then(function(response) { return response.json(); })
    .then(function(result) {
        if (result.success) {
            showStatus('training-status', '\u2705 \u6a21\u578b\u8bad\u7ec3\u5b8c\u6210\uff01', 'success');
            modelResultsData = result.model_results;
            modelColors = result.model_colors || MODEL_COLORS;
            bestModelName = result.best_model;
            document.getElementById('module-2').style.display = 'block';
            document.getElementById('best-model-name').textContent = result.best_model;
            var tbody = document.getElementById('model-table-body');
            tbody.innerHTML = '';
            for (var i = 0; i < MODEL_NAMES.length; i++) {
                var modelName = MODEL_NAMES[i];
                var metrics = result.model_results[modelName];
                if (metrics) {
                    var row = document.createElement('tr');
                    var rowColor = getColor(modelName);
                    row.innerHTML = '<td><strong style="color:' + rowColor + '">' + modelName + '</strong></td>' +
                        '<td>' + formatNum(metrics.Accuracy) + '</td>' +
                        '<td>' + formatNum(metrics.Precision) + '</td>' +
                        '<td>' + formatNum(metrics.Recall) + '</td>' +
                        '<td>' + formatNum(metrics.F1_Score) + '</td>' +
                        '<td>' + formatNum(metrics.AUC_ROC) + '</td>' +
                        '<td>' + formatNum(metrics.AUPRC) + '</td>';
                    tbody.appendChild(row);
                }
            }
            document.getElementById('module-3').style.display = 'block';
            document.getElementById('charts-section-1').style.display = 'block';
            setTimeout(function() { createAllCharts(); }, 300);
            scrollToSection('module-2');
        } else {
            showStatus('training-status', '\u274c ' + result.error, 'error');
        }
    })
    .catch(function(error) {
        showStatus('training-status', '\u274c \u4e0a\u4f20\u5931\u8d25: ' + error.message, 'error');
    });
}

function uploadDatabase() {
    var fileInput = document.getElementById('database-file');
    var file = fileInput.files[0];
    if (!file) { showStatus('screening-status', '\u8bf7\u9009\u62e9\u5f85\u7b5b\u9009\u6570\u636e\u5e93\u6587\u4ef6', 'error'); return; }
    showStatus('screening-status', '\u23f3 \u6b63\u5728\u8fdb\u884c\u6570\u636e\u5e93\u7b5b\u9009\uff0c\u8bf7\u7a0d\u5019...', 'loading');
    var formData = new FormData();
    formData.append('file', file);
    fetch('/api/upload-screening-database', { method: 'POST', body: formData })
    .then(function(response) { return response.json(); })
    .then(function(result) {
        if (result.success) {
            showStatus('screening-status', '\u2705 \u7b5b\u9009\u5b8c\u6210\uff01', 'success');
            screeningResultsData = result.screening_results;
            var sd = result.screening_results;
            document.getElementById('total-compounds').textContent = sd.total_compounds;
            document.getElementById('active-compounds').textContent = sd.active_compounds;
            document.getElementById('inactive-compounds').textContent = sd.inactive_compounds;
            document.getElementById('flavonoid-count').textContent = sd.flavonoid_in_top50;
            document.getElementById('screening-results').style.display = 'block';
            document.getElementById('charts-section-2').style.display = 'block';
            setTimeout(function() { createScreeningCharts(); }, 300);
            scrollToSection('module-3');
        } else {
            showStatus('screening-status', '\u274c ' + result.error, 'error');
        }
    })
    .catch(function(error) {
        showStatus('screening-status', '\u274c \u7b5b\u9009\u5931\u8d25: ' + error.message, 'error');
    });
}

function downloadFile(filename) {
    fetch('/api/download-file', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: filename })
    })
    .then(function(response) { if (response.ok) return response.blob(); })
    .then(function(blob) {
        if (blob) {
            var url = window.URL.createObjectURL(blob);
            var a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        }
    });
}

function resetSystem() {
    fetch('/api/clear-session', { method: 'POST' })
    .then(function() {
        document.getElementById('training-file').value = '';
        document.getElementById('database-file').value = '';
        document.getElementById('training-status').innerHTML = '';
        document.getElementById('screening-status').innerHTML = '';
        document.getElementById('module-2').style.display = 'none';
        document.getElementById('module-3').style.display = 'none';
        document.getElementById('screening-results').style.display = 'none';
        document.getElementById('charts-section-1').style.display = 'none';
        document.getElementById('charts-section-2').style.display = 'none';
        modelResultsData = null;
        screeningResultsData = null;
        bestModelName = null;
        scrollToSection('home');
    });
}

function createAllCharts() {
    if (!modelResultsData) return;
    for (var i = 0; i < MODEL_NAMES.length; i++) {
        var modelName = MODEL_NAMES[i];
        if (modelResultsData[modelName]) {
            createIndividualModelCharts(modelName, modelResultsData[modelName], getColor(modelName));
        }
    }
    createPerformanceComparisonChart();
    createROCComparisonChart();
    createRadarChart();
    createTrainingTimeChart();
}

function createIndividualModelCharts(modelName, metrics, color) {
    var prefix = MODEL_KEY[modelName];

    // 1. Main Performance Metrics bar chart
    var metricsEl = document.getElementById(prefix + '-metrics');
    if (metricsEl) {
        var chart = echarts.init(metricsEl);
        var perfMetrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'Specificity'];
        var perfValues = perfMetrics.map(function(m) { return metrics[m] || 0; });

        chart.setOption({
            title: { text: 'Main Performance Metrics', left: 'center', top: 5, textStyle: { fontSize: 13, fontWeight: 'bold' } },
            tooltip: { trigger: 'axis', formatter: function(params) { return params[0].name + ': ' + params[0].value.toFixed(4); } },
            grid: { left: '18%', right: '8%', top: '22%', bottom: '15%' },
            xAxis: { type: 'category', data: perfMetrics, axisLabel: { rotate: 30, fontSize: 9 }, axisTick: { alignWithLabel: true } },
            yAxis: { type: 'value', name: 'Score', min: 0, max: 1, splitLine: { lineStyle: { type: 'dashed', color: '#e0e0e0' } } },
            series: [{
                type: 'bar',
                data: perfValues,
                itemStyle: { color: color, opacity: 0.7 },
                barWidth: '50%',
                label: { show: true, position: 'top', formatter: function(p) { return p.data.toFixed(3); }, fontSize: 9, color: '#333' }
            }]
        });
    }

    // 2. ROC curve with model-specific color fill
    var rocEl = document.getElementById(prefix + '-roc');
    if (rocEl) {
        var chart = echarts.init(rocEl);
        var rocData = metrics.roc_curve || { fpr: [0, 1], tpr: [0, 1] };
        var aucScore = metrics.AUC_ROC || 0.5;
        var rocPoints = [];
        for (var j = 0; j < rocData.tpr.length; j++) {
            rocPoints.push([rocData.fpr[j], rocData.tpr[j]]);
        }

        chart.setOption({
            title: { text: 'ROC Curve (AUC=' + aucScore.toFixed(3) + ')', left: 'center', top: 5, textStyle: { fontSize: 13, fontWeight: 'bold' } },
            tooltip: { trigger: 'axis', formatter: function(params) { return 'FPR: ' + params[0].value[0].toFixed(3) + '<br>TPR: ' + params[0].value[1].toFixed(3); } },
            grid: { left: '12%', right: '8%', top: '22%', bottom: '12%' },
            xAxis: { type: 'value', name: 'False Positive Rate', nameLocation: 'center', nameGap: 25, min: 0, max: 1, splitLine: { lineStyle: { type: 'dashed', color: '#e0e0e0' } } },
            yAxis: { type: 'value', name: 'True Positive Rate', nameLocation: 'center', nameGap: 30, min: 0, max: 1, splitLine: { lineStyle: { type: 'dashed', color: '#e0e0e0' } } },
            series: [{
                type: 'line',
                data: rocPoints,
                smooth: true,
                showSymbol: false,
                lineStyle: { width: 2, color: color },
                areaStyle: { color: color, opacity: 0.2 },
                markLine: {
                    symbol: 'none',
                    data: [[{ coord: [0, 0] }, { coord: [1, 1] }]],
                    lineStyle: { type: 'dashed', color: '#999', width: 1 }
                }
            }]
        });
    }

    // 3. Confusion Matrix heatmap
    var cmEl = document.getElementById(prefix + '-cm');
    if (cmEl) {
        var chart = echarts.init(cmEl);
        var cm = metrics.confusion_matrix || [[0, 0], [0, 0]];
        var maxVal = Math.max.apply(null, cm.flat()) || 1;

        chart.setOption({
            title: { text: 'Confusion Matrix', left: 'center', top: 5, textStyle: { fontSize: 13, fontWeight: 'bold' } },
            tooltip: { position: 'top' },
            grid: { left: '18%', right: '12%', top: '22%', bottom: '18%' },
            xAxis: { type: 'category', data: ['Pred:0', 'Pred:1'], splitArea: { show: true } },
            yAxis: { type: 'category', data: ['True:0', 'True:1'], splitArea: { show: true } },
            visualMap: {
                min: 0, max: maxVal, calculable: true, orient: 'horizontal', left: 'center', bottom: '2%',
                inRange: { color: ['#e8f4fd', '#b3d9f2', '#5baed6', '#2171b5', '#084594'] },
                text: [maxVal, 0], textStyle: { fontSize: 9 }
            },
            series: [{
                type: 'heatmap',
                data: [[0, 0, cm[0][0]], [0, 1, cm[0][1]], [1, 0, cm[1][0]], [1, 1, cm[1][1]]],
                label: { show: true, fontSize: 12, fontWeight: 'bold' },
                emphasis: { itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.3)' } }
            }]
        });
    }

    // 4. Training Time & Cross-Validation (dual axis like python.py)
    var timeEl = document.getElementById(prefix + '-time');
    if (timeEl) {
        var chart = echarts.init(timeEl);
        var trainTime = metrics['Training_Time(s)'] || 0;
        var cvMean = metrics.CV_F1_Mean || 0;
        var cvStd = metrics.CV_F1_Std || 0;

        chart.setOption({
            title: { text: 'Training Time & Cross-Validation', left: 'center', top: 5, textStyle: { fontSize: 13, fontWeight: 'bold' } },
            tooltip: { trigger: 'axis' },
            grid: { left: '12%', right: '15%', top: '22%', bottom: '15%' },
            xAxis: { type: 'category', data: ['Training Time', 'Cross-Validation'] },
            yAxis: [
                { type: 'value', name: 'Time(s)', nameLocation: 'center', nameGap: 35, axisLabel: { color: color }, splitLine: { lineStyle: { type: 'dashed', color: '#e0e0e0' } } },
                { type: 'value', name: 'F1 Score', min: 0, max: 1, nameLocation: 'center', nameGap: 40, axisLabel: { color: '#d62728' }, splitLine: { show: false } }
            ],
            series: [
                {
                    type: 'bar', data: [trainTime, null],
                    itemStyle: { color: color, opacity: 0.6 }, barWidth: '30%',
                    label: { show: true, position: 'top', formatter: function(p) { return p.data != null ? p.data.toFixed(3) + 's' : ''; }, fontSize: 9, color: color }
                },
                {
                    type: 'scatter', data: [null, cvMean], yAxisIndex: 1,
                    symbolSize: 12, itemStyle: { color: '#d62728' },
                    label: { show: true, position: 'top', formatter: function(p) { return p.data != null ? p.data.toFixed(3) + '\u00b1' + cvStd.toFixed(3) : ''; }, fontSize: 9, color: '#d62728' }
                }
            ]
        });
    }
}

// Performance comparison chart - grouped by metric (matching python.py create_performance_bar_chart)
function createPerformanceComparisonChart() {
    var chartDom = document.getElementById('model-performance-chart');
    if (!chartDom) return;
    var chart = echarts.init(chartDom);
    var metricsList = ['F1_Score', 'AUC_ROC', 'Accuracy', 'Precision', 'Recall'];
    var colors = MODEL_NAMES.map(function(n) { return getColor(n); });

    var series = [];
    for (var i = 0; i < MODEL_NAMES.length; i++) {
        var data = [];
        for (var m = 0; m < metricsList.length; m++) {
            var val = modelResultsData[MODEL_NAMES[i]] ? (modelResultsData[MODEL_NAMES[i]][metricsList[m]] || 0) : 0;
            data.push(val);
        }
        series.push({
            name: MODEL_NAMES[i],
            type: 'bar',
            data: data,
            itemStyle: { color: colors[i], opacity: 0.8 }
        });
    }

    chart.setOption({
        title: { text: 'Model Performance Metrics Comparison', left: 'center', textStyle: { fontSize: 14, fontWeight: 'bold' } },
        tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
        legend: { top: 30, data: MODEL_NAMES, type: 'scroll' },
        grid: { left: '8%', right: '5%', top: '22%', bottom: '12%' },
        xAxis: { type: 'category', data: metricsList, axisLabel: { rotate: 30, fontSize: 10 } },
        yAxis: { type: 'value', name: 'Score', min: 0, max: 1, splitLine: { lineStyle: { type: 'dashed', color: '#e0e0e0' } } },
        series: series
    });
}

// ROC comparison chart - no fill, only lines (matching python.py create_roc_comparison_chart)
function createROCComparisonChart() {
    var chartDom = document.getElementById('roc-comparison-chart');
    if (!chartDom) return;
    var chart = echarts.init(chartDom);

    var series = [];
    for (var i = 0; i < MODEL_NAMES.length; i++) {
        var name = MODEL_NAMES[i];
        if (modelResultsData[name] && modelResultsData[name].roc_curve) {
            var roc = modelResultsData[name].roc_curve;
            var auc = modelResultsData[name].AUC_ROC || 0.5;
            var data = [];
            for (var j = 0; j < roc.tpr.length; j++) {
                data.push([roc.fpr[j], roc.tpr[j]]);
            }
            series.push({
                name: name + ' (AUC=' + auc.toFixed(3) + ')',
                type: 'line',
                data: data,
                smooth: true,
                showSymbol: false,
                lineStyle: { width: 2 },
                itemStyle: { color: getColor(name) }
            });
        }
    }
    series.push({
        name: 'Random',
        type: 'line',
        data: [[0, 0], [1, 1]],
        lineStyle: { type: 'dashed', width: 1, color: '#999' },
        showSymbol: false
    });

    chart.setOption({
        title: { text: 'ROC Curves Comparison', left: 'center', textStyle: { fontSize: 14, fontWeight: 'bold' } },
        tooltip: { trigger: 'axis' },
        legend: { top: 30, type: 'scroll', data: MODEL_NAMES.map(function(n) { return n + ' (AUC=' + (modelResultsData[n] ? (modelResultsData[n].AUC_ROC || 0.5).toFixed(3) : '0.500') + ')'; }) },
        grid: { left: '10%', right: '5%', top: '22%', bottom: '12%' },
        xAxis: { type: 'value', name: 'False Positive Rate', nameLocation: 'center', nameGap: 30, min: 0, max: 1, splitLine: { lineStyle: { type: 'dashed', color: '#e0e0e0' } } },
        yAxis: { type: 'value', name: 'True Positive Rate', nameLocation: 'center', nameGap: 35, min: 0, max: 1.05, splitLine: { lineStyle: { type: 'dashed', color: '#e0e0e0' } } }
    });
    chart.setOption({ series: series });
}

// Radar chart - no fill, only lines (matching python.py create_radar_chart)
function createRadarChart() {
    var chartDom = document.getElementById('radar-chart');
    if (!chartDom) return;
    var chart = echarts.init(chartDom);
    var indicators = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'Specificity'];

    var series = [];
    for (var i = 0; i < MODEL_NAMES.length; i++) {
        var name = MODEL_NAMES[i];
        if (modelResultsData[name]) {
            var values = [];
            for (var j = 0; j < indicators.length; j++) {
                values.push(modelResultsData[name][indicators[j]] || 0);
            }
            series.push({
                name: name,
                type: 'radar',
                data: [{ value: values, name: name }],
                lineStyle: { width: 2, color: getColor(name), opacity: 0.7 },
                itemStyle: { color: getColor(name) },
                symbol: 'circle',
                symbolSize: 4
            });
        }
    }

    chart.setOption({
        title: { text: 'Model Performance Radar Chart', left: 'center', textStyle: { fontSize: 14, fontWeight: 'bold' } },
        tooltip: {},
        legend: { top: 30, data: MODEL_NAMES },
        radar: {
            indicator: indicators.map(function(ind) { return { name: ind, max: 1 }; }),
            radius: '60%',
            splitLine: { lineStyle: { color: '#ddd' } },
            splitArea: { areaStyle: { color: ['rgba(114,172,239,0.05)', 'rgba(114,172,239,0.1)'] } },
            axisLine: { lineStyle: { color: '#ccc' } }
        },
        series: series
    });
}

// Training time chart (matching python.py create_training_time_chart)
function createTrainingTimeChart() {
    var chartDom = document.getElementById('training-time-chart');
    if (!chartDom) return;
    var chart = echarts.init(chartDom);
    var times = [];
    var colors = [];
    for (var i = 0; i < MODEL_NAMES.length; i++) {
        times.push(modelResultsData[MODEL_NAMES[i]] ? (modelResultsData[MODEL_NAMES[i]]['Training_Time(s)'] || 0) : 0);
        colors.push(getColor(MODEL_NAMES[i]));
    }

    chart.setOption({
        title: { text: 'Model Training Time Comparison', left: 'center', textStyle: { fontSize: 14, fontWeight: 'bold' } },
        tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
        grid: { left: '10%', right: '40%', top: '22%', bottom: '12%' },
        xAxis: { type: 'category', data: MODEL_NAMES, axisLabel: { rotate: 30 }, splitLine: { show: false } },
        yAxis: { type: 'value', name: 'Time(s)', splitLine: { lineStyle: { type: 'dashed', color: '#e0e0e0' } } },
        series: [
            {
                type: 'bar', data: times.map(function(t, i) { return { value: t, itemStyle: { color: colors[i], opacity: 0.8 } }; }),
                label: { show: true, position: 'top', formatter: function(p) { return p.value.toFixed(3) + 's'; }, fontSize: 9 }
            },
            {
                type: 'pie',
                center: ['78%', '55%'],
                radius: ['20%', '40%'],
                data: times.map(function(t, i) { return { value: t, name: MODEL_NAMES[i], itemStyle: { color: colors[i] } }; }),
                label: { fontSize: 8 },
                emphasis: { itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.2)' } }
            }
        ]
    });
}

// Screening charts (matching python.py create_screening_results_chart)
function createScreeningCharts() {
    if (!screeningResultsData) return;
    var sd = screeningResultsData;

    // 1. Active/Inactive pie chart
    var pieEl = document.getElementById('screen-pie');
    if (pieEl) {
        var chart = echarts.init(pieEl);
        chart.setOption({
            title: { text: 'Active Compound Distribution', left: 'center', top: 5, textStyle: { fontSize: 13, fontWeight: 'bold' } },
            tooltip: { trigger: 'item', formatter: '{b}: {c} ({d}%)' },
            legend: { bottom: 5, data: ['Active', 'Inactive'], textStyle: { fontSize: 10 } },
            series: [{
                type: 'pie', radius: ['25%', '55%'], center: ['50%', '50%'],
                data: [
                    { value: sd.active_compounds, name: 'Active', itemStyle: { color: '#ff6b6b' } },
                    { value: sd.inactive_compounds, name: 'Inactive', itemStyle: { color: '#4ecdc4' } }
                ],
                label: { fontSize: 10, formatter: '{b}\n{d}%' },
                emphasis: { itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.2)' } }
            }]
        });
    }

    // 2. Probability distribution histogram
    var histEl = document.getElementById('screen-hist');
    if (histEl) {
        var chart = echarts.init(histEl);
        var bins = sd.probability_bins || [];
        var labels = bins.map(function(_, i) { return (i * 0.1).toFixed(1); });
        chart.setOption({
            title: { text: 'Active Probability Distribution', left: 'center', top: 5, textStyle: { fontSize: 13, fontWeight: 'bold' } },
            tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
            grid: { left: '12%', right: '8%', top: '22%', bottom: '15%' },
            xAxis: { type: 'category', data: labels, axisLabel: { rotate: 45, fontSize: 9 }, name: 'Probability', nameLocation: 'center', nameGap: 30 },
            yAxis: { type: 'value', name: 'Count', splitLine: { lineStyle: { type: 'dashed', color: '#e0e0e0' } } },
            series: [{
                type: 'bar', data: bins,
                itemStyle: { color: '#118ab2', opacity: 0.7 },
                label: { show: true, position: 'top', fontSize: 8 }
            }]
        });
    }

    // 3. Rank vs Probability scatter
    var rankEl = document.getElementById('screen-rank');
    if (rankEl) {
        var chart = echarts.init(rankEl);
        var topResults = sd.top_results || [];
        var data = topResults.slice(0, 50).map(function(r) { return [r.rank, r.probability]; });
        chart.setOption({
            title: { text: 'Rank vs Probability (Top50)', left: 'center', top: 5, textStyle: { fontSize: 13, fontWeight: 'bold' } },
            tooltip: { trigger: 'axis', formatter: function(params) { return 'Rank: ' + params[0].value[0] + '<br>Probability: ' + params[0].value[1].toFixed(3); } },
            grid: { left: '12%', right: '8%', top: '22%', bottom: '12%' },
            xAxis: { type: 'value', name: 'Rank', nameLocation: 'center', nameGap: 25, splitLine: { lineStyle: { type: 'dashed', color: '#e0e0e0' } } },
            yAxis: { type: 'value', name: 'Probability', min: 0, max: 1, nameLocation: 'center', nameGap: 30, splitLine: { lineStyle: { type: 'dashed', color: '#e0e0e0' } } },
            series: [{
                type: 'scatter', data: data,
                symbolSize: 6,
                itemStyle: { color: '#ef476f', opacity: 0.6 },
                emphasis: { itemStyle: { opacity: 1, shadowBlur: 5 } }
            }]
        });
    }

    // 4. Flavonoid distribution pie chart
    var flavEl = document.getElementById('screen-flav');
    if (flavEl) {
        var chart = echarts.init(flavEl);
        var flavCount = sd.flavonoid_in_top50;
        var nonFlavCount = 50 - flavCount;
        chart.setOption({
            title: { text: 'Flavonoid Distribution (Top50)', left: 'center', top: 5, textStyle: { fontSize: 13, fontWeight: 'bold' } },
            tooltip: { trigger: 'item', formatter: '{b}: {c} ({d}%)' },
            legend: { bottom: 5, data: ['Flavonoid', 'Non-Flavonoid'], textStyle: { fontSize: 10 } },
            series: [{
                type: 'pie', radius: ['25%', '55%'], center: ['50%', '50%'],
                data: [
                    { value: flavCount, name: 'Flavonoid', itemStyle: { color: '#ffd166' } },
                    { value: nonFlavCount, name: 'Non-Flavonoid', itemStyle: { color: '#06d6a0' } }
                ],
                label: { fontSize: 10, formatter: '{b}\n{d}%' },
                emphasis: { itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.2)' } }
            }]
        });
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    var navLinks = document.querySelectorAll('.nav-links a');
    for (var i = 0; i < navLinks.length; i++) {
        navLinks[i].addEventListener('click', function(e) {
            e.preventDefault();
            scrollToSection(this.getAttribute('href').substring(1));
        });
    }
});
