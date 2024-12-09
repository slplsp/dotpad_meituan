// 절대 건들지 마세요 1~58
import { DotPad } from "../static/DotPad_Class.js"
let myDotPad = new DotPad();


document.getElementById("connectbutton").addEventListener("click", onConnectButtonClick, false);
document.getElementById("disconnectbutton").addEventListener("click", onDisconnectButtonClick, false);




//블루투스 연결을 위한 함수들
async function onConnectButtonClick() {
  try {
    console.log('Requesting Bluetooth Device...');
    //console.log('with ' + JSON.stringify(options));
    myDotPad.connect();

  } catch (error) {
    console.log('> Error:' + error);
  }
}
async function onDisconnectButtonClick() {
  try {
    myDotPad.disconnect();
  }
  catch (error) {
    console.log('> Error:' + error);
  }
}


//닷 패드에 보내기 위해 2진수를 16진수로 변환
function hexa(a) {
  let b = parseInt(a, 2);
  b = b.toString(16);
  return b;
}

function trans_hex_pad(a) {
  let trans_list = '';
  const J = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36];
  for (let j of J) {

    let s = '';
    let ss = '';
    for (let i = 0; i < a[0].length - 1; i += 2) {
      s = '' + a[j + 3][i] + a[j + 2][i] + a[j + 1][i] + a[j][i];
      ss = '' + a[j + 3][i + 1] + a[j + 2][i + 1] + a[j + 1][i + 1] + a[j][i + 1];

      trans_list += hexa(ss) + hexa(s);
    }

  }
  return trans_list;
}

document.getElementById("sendbutton").addEventListener("click", onS4ButtonClick, false);




async function onS4ButtonClick() {
    try {
        // Fetch the 2D array from backend
        var response = await fetch('/image2text/get-array/');
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();

        // Ensure the backend response contains the array
        if (!data || !Array.isArray(data.array)) {
            throw new Error("Invalid data format received from the backend.");
        }

        const array2D = data.array;

        // Process the 2D array as needed
        const F1 = [];
        F1.push(
            "0000000000000000000000000000000000000000" +
            trans_hex_pad(array2D)
        );

        // Send the data to DotPad
        myDotPad.send(F1[0]);
        console.log("Data sent to DotPad successfully.");
    } catch (error) {
        console.error("Failed to fetch and send data to DotPad:", error);
    }
}

