window.dccFunctions = window.dccFunctions || {};
window.dccFunctions.hourMinute = function(value) {
    // value is a number, e.g., 10.75
    var hours = Math.floor(value);
    var minutes = Math.round((value - hours) * 60);
    // Pad with zeros if needed
    var hoursStr = hours.toString().padStart(2, '0');
    var minutesStr = minutes.toString().padStart(2, '0');
    return hoursStr + ":" + minutesStr;
}